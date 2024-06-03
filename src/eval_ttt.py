import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, get_scheduler
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from peft import LoraConfig, TaskType, get_peft_model

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from eval_llama import truncate_input, truncate_by_tokens, chunk_generate
from argparse import ArgumentParser
from tqdm import tqdm
import gc

MAX_POSITION_ID = 128 * 1024  # Determined by the model
TRUNCATE_LEN = 128 * 1024

def parse_args() -> Any:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="../results", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        choices=["gpt4", "yarn-mistral", "kimi", "claude2", "rwkv", "yi-6b-200k", "yi-34b-200k", "chatglm3", "llama2"],
        default="gpt4",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    
    p.add_argument(
        "--lora_rank", type=int, default=64, help="Rank of the LORA approximation."
    )
    p.add_argument(
        "--lora_alpha", type=float, default=64.0, help="Alpha of the LORA approximation."
    )
    p.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Dropout of the LORA approximation."
    )
    p.add_argument(
        "--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing."
    )
    p.add_argument(
        "--training_ctx_len", type=int, default=3072, help="The context length for training."
    )
    p.add_argument(
        "--stride", type=int, default=1024, help="The stride for training."
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=1, help="The number of training epochs."
    )
    p.add_argument(
        "--learning_rate", type=float, default=5e-5, help="The learning rate."
    )
    p.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay."
    )
    p.add_argument(
        "--warmup_steps", type=int, default=2, help="The warmup steps."
    )
    p.add_argument(
        "--lr_scheduler_type", type=str, default="constant_with_warmup", help="The learning rate scheduler type."
    )
    return p.parse_args()
    

############# model utils #############
def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    linear_class = torch.nn.Linear
    lora_module_names = set([])

    for name, module in model.named_modules():
        if isinstance(module, linear_class):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def load_model(
    model_name: str = "../../../yarn-llama2-7b-128k",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2")
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=find_all_linear_names(model=model)
        )
        model.enable_input_require_grads()
        model = get_peft_model(model=model, peft_config=lora_config)
        model.print_trainable_parameters()

    print("Time taken:", round(time.time() - start_time))
    return model, tok  # type: ignore


############# optim utils #############
def create_lr_scheduler(
        num_training_steps: int, optimizer: torch.optim.Optimizer, args: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    return lr_scheduler


def create_optimizer(model: PreTrainedModel, args: Any) -> torch.optim.Optimizer:
    decay_parameters = get_decay_parameter_names(model=model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0
        }
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-7
    )


def get_decay_parameter_names(model: PreTrainedModel) -> List[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def chunk_train(
    model,
    tok,
    optim,
    texts: List[str],
    max_tokens: int,
    training_ctx_len: int,
    stride: int,
    num_train_epochs: int,
):
    """
    This function is structurally similar to `chunk_generate`.
    Moreover, it is more memory efficient. Because we do not have to build the KV cache here.
    Rather we have to use extra memory for LORA parameters and their gradients.  
    input_ids: (b, n)
    labels: (b, n)
    """
    model.train()
    inputs = tok(texts, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)  # type: ignore
    input_ids: Tensor = inputs.input_ids  # (b, n)
    seq_len = input_ids.shape[-1]

    num_training_steps = (seq_len - training_ctx_len) // stride
    lr_scheduler = create_lr_scheduler(
        num_training_steps=num_training_steps, optimizer=optim, args=args
    )
    pbar = tqdm(range(0, seq_len, stride))
    for middle in pbar:
        start = max(0, middle - training_ctx_len)
        end = min(seq_len, middle + stride)
        chunk_input_ids = input_ids[:, start:end]
        chunk_labels = chunk_input_ids.clone()
        chunk_labels[:, :(start - middle)] = -100
        chunk_input_ids = chunk_input_ids.to(model.device)
        chunk_labels = chunk_labels.to(model.device)
        for i in range(num_train_epochs):
            optim.zero_grad()
            loss = model(input_ids=chunk_input_ids, labels=chunk_labels).loss
            loss.backward()
            optim.step()
            pbar.set_description(f"Loss: {loss.item()}")
        lr_scheduler.step()
    model.eval()
    return model


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    args: Any = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    
    optim = create_optimizer(model, args)
    model = chunk_train(
        model,
        tok,
        optim,
        [input_text],
        max_tokens=max_tokens,
        training_ctx_len=args.training_ctx_len,
        stride=args.stride,
        num_train_epochs=args.num_train_epochs,
    ) 
    
    del optim
    gc.collect()
    torch.cuda.empty_cache()

    output = chunk_generate(
        model,
        tok,
        [input_text],
        max_tokens=max_tokens,
        chunk_size=128,
        verbose=verbose,
    )[0]
    print("Chunked generation:", output)
    return output


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    
    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_ttt_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_ttt_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")

        # create the model from scratch for every example
        model, tok = load_model(args.model_path)
        answer = get_answer(eg, data_name)
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose, args=args
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
