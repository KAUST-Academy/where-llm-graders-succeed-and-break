#!/usr/bin/env python3
"""QLoRA fine-tune one Qwen model to predict exam marks.

Trains on the conversational prompt/completion JSONL from
``build_finetune_data.py``. Loss is computed on the assistant completion only
(the short ``{"score": .., "bonus": ..}`` JSON) -- the long grading prompt is
masked, so the model is supervised purely on the marks.

QLoRA: the base model is loaded in 4-bit (nf4, double-quant, bf16 compute) and
only LoRA adapters are trained, so each of Qwen2.5-Coder-7B / -14B and
Qwen3-Coder-30B-A3B fits on a single A100 (80 GB comfortably; 40 GB for the
7B/14B and the 30B with a reduced --max-seq-len).

Run on a GPU node (same place you run serve_qwen.sh), e.g.:
    python finetune/train_lora.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --data finetune/data/train.jsonl \
        --output-dir finetune/adapters/qwen2.5-coder-7b
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def decoder_layer_cls_name(model) -> str:
    """Name of the repeated transformer block, for FSDP's auto-wrap policy.

    Read off the live module tree rather than hardcoded per model, so this works
    for Qwen2.5-Coder (Qwen2DecoderLayer) and Qwen3-Coder-MoE (Qwen3MoeDecoderLayer)
    alike. Wrapping at the block boundary makes each all-gather one block's worth
    of weights (~1.3 GB for the 30B) instead of the whole model.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        raise SystemExit(
            "Could not find the decoder layers (model.model.layers) to build an "
            "FSDP auto-wrap policy for this architecture."
        )
    return type(layers[0]).__name__


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HuggingFace model id (base).")
    ap.add_argument("--data", type=Path, default=Path("finetune/data/train.jsonl"))
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to save the trained LoRA adapter.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-device micro-batch (prompts are long; keep small).")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=12288,
                    help="Truncation length for prompt+completion. Clears every "
                         "current row (longest prompt ~10.3k tok) so the tiny "
                         "completion is never truncated away. Drop to ~6144 on a "
                         "40 GB GPU (long prompts then lose their tail).")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="all-linear",
                    help="LoRA target modules ('all-linear' or comma-separated names).")
    ap.add_argument("--skip-kbit-upcast", action="store_true",
                    help="With --4bit on an MoE: skip PEFT's fp32 upcast of "
                         "non-quantized weights. bitsandbytes leaves MoE experts in "
                         "bf16, so the upcast doubles ~90%% of the model and OOMs. "
                         "Required for Qwen3-Coder-30B-A3B in 4-bit.")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load the base model in bf16 instead of 4-bit (needs more VRAM).")
    ap.add_argument("--fsdp", action="store_true",
                    help="Shard one bf16 model across the ranks (FSDP full_shard) instead "
                         "of giving every rank a full copy (DDP). Needed for the 30B in "
                         "bf16: 61 GB of weights / 4 GPUs fits, 61 GB per GPU does not. "
                         "Implies --no-4bit; only meaningful under torchrun.")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="Attention impl. Default sdpa (flash-attn not installed in ali-env).")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="0 = save only at the end of each epoch.")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="CHECKPOINT",
                    help="Resume an interrupted run (e.g. the SLURM job expired "
                         "mid-training). Bare --resume picks the latest "
                         "checkpoint-* in --output-dir; or pass an explicit path. "
                         "Optimizer/scheduler/RNG state are restored from the "
                         "checkpoint, so the resumed run continues the same LR "
                         "schedule rather than restarting it.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. Run this on a GPU node (the same place you "
            "launch serve_qwen.sh)."
        )

    print(f"== QLoRA fine-tune ==")
    print(f"  model       : {args.model}")
    print(f"  data        : {args.data}")
    print(f"  output      : {args.output_dir}")
    print(f"  seed        : {args.seed}  epochs={args.epochs}  lr={args.lr}")
    print(f"  4-bit       : {not args.no_4bit}  max_seq_len={args.max_seq_len}")
    print(f"  sharding    : {'FSDP full_shard' if args.fsdp else 'DDP (full replica per rank)'}")

    # FSDP has to be announced via env BEFORE from_pretrained: transformers only
    # skips materialising weights on the non-zero ranks when ACCELERATE_USE_FSDP
    # and FSDP_CPU_RAM_EFFICIENT_LOADING are both set *and* the process group is
    # already up (integrations/fsdp.py::is_fsdp_enabled). SFTConfig sets the
    # second one itself, but only at construction time -- far too late, by which
    # point every rank would have loaded its own full copy into CPU RAM (4 x
    # 61 GB for the 30B, on a 376 GB node).
    if args.fsdp:
        if not args.no_4bit:
            raise SystemExit(
                "--fsdp requires --no-4bit: FSDP shards plain bf16 parameters, and "
                "transformers skips the CPU-RAM-efficient path entirely for quantized "
                "models. Pick one -- bf16+FSDP (better quality) or 4-bit QLoRA."
            )
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
        PartialState()  # init_process_group, so is_local_dist_rank_0() can answer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Place each process's model replica on its own GPU. Single-GPU (plain
    # `python`) -> process_index 0; multi-GPU (`torchrun --nproc_per_node=N`)
    # -> ranks 0..N-1 each load a full copy for data-parallel (DDP) training.
    # Under FSDP there is no replica to place: device_map must stay None so FSDP
    # can shard the parameters across the ranks itself.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map=None if args.fsdp else {"": PartialState().process_index},
        trust_remote_code=True,
        attn_implementation=args.attn,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        # prepare_model_for_kbit_training() upcasts every non-Params4bit weight to
        # fp32. On a dense model that's just norms/embeddings. On an MoE it also
        # hits the expert weights -- bitsandbytes leaves those unquantized -- so
        # ~90% of a Qwen3-MoE gets DOUBLED (bf16 -> fp32) and OOMs a 80 GB card
        # before step 0. Skip the upcast there and only enable checkpointing +
        # input grads, which is all we actually need for LoRA.
        if args.skip_kbit_upcast:
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model.enable_input_require_grads()
            for p in model.parameters():
                p.requires_grad = False
            print("  kbit prep  : upcast SKIPPED (MoE-safe); grad-ckpt + input grads on")
        else:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )

    target_modules = (
        "all-linear" if args.target_modules == "all-linear"
        else [m.strip() for m in args.target_modules.split(",") if m.strip()]
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    dataset = load_dataset("json", data_files=str(args.data), split="train")
    print(f"  train rows  : {len(dataset)}")

    fsdp_kwargs = {}
    if args.fsdp:
        wrap_cls = decoder_layer_cls_name(model)
        print(f"  fsdp wrap   : {wrap_cls}")
        fsdp_kwargs = dict(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": [wrap_cls],
                # LoRA leaves frozen and trainable params side by side inside one
                # wrapped block; FSDP only tolerates that mix with use_orig_params.
                "use_orig_params": True,
                # rank 0 holds the real weights and broadcasts them to the ranks
                # that came up on meta -- required by cpu_ram_efficient_loading.
                "sync_module_states": True,
                "cpu_ram_efficient_loading": True,
                "version": 1,
            },
        )

    sft_config = SFTConfig(
        **fsdp_kwargs,
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        # DDP: LoRA freezes most params, so tell DDP not to look for grads on
        # them (avoids the "unused parameter" error and a sync slowdown).
        ddp_find_unused_parameters=False,
        # Conversational prompt/completion data -> loss on the assistant turn only.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --resume: bare flag -> newest checkpoint-N in output_dir; else an explicit path.
    resume = args.resume
    if resume == "auto":
        ckpts = sorted(
            (p for p in args.output_dir.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            # Fresh start instead of a hard error: chain scripts set RESUME=1
            # globally, and a training that has never begun has no checkpoints --
            # that is not a mistake, there is simply nothing to resume.
            print(f"--resume: no checkpoint-* in {args.output_dir}; "
                  "starting from scratch.")
            resume = None
        else:
            resume = str(ckpts[-1])
    if resume:
        if not Path(resume, "trainer_state.json").exists():
            raise SystemExit(
                f"--resume: {resume} has no trainer_state.json; it is not a "
                "resumable checkpoint."
            )
        print(f"  RESUMING from {resume} (optimizer/scheduler/RNG restored)")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nSaved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
