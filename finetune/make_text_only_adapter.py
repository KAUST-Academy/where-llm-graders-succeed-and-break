#!/usr/bin/env python3
"""Write a text-only copy of a Gemma-4 LoRA adapter for vLLM serving.

The Gemma-4 adapters were trained with all-linear targets over the full
multimodal checkpoint, so their safetensors carry LoRA tensors for the audio
tower, the vision tower and the two modality embedders (494 of 1,184 tensors)
as well as the language model (686). vLLM's LoRA loader accepts only the
language-model projections and rejects the whole adapter ("expected target
modules in {q_proj, ..., per_layer_projection} but received
model.audio_tower..."). Text-only grading never exercises the audio/vision
paths, so dropping those tensors leaves every text output unchanged.

    python finetune/make_text_only_adapter.py finetune/adapters/gemma-both finetune/adapters_textonly/gemma-both
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
tensors = load_file(src / "adapter_model.safetensors")
keep = {k: v for k, v in tensors.items() if ".language_model." in k}
dropped = sorted({k.split(".lora_")[0].split("base_model.model.model.")[1].split(".")[0] for k in tensors if k not in keep})
save_file(keep, dst / "adapter_model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(src / "adapter_config.json"))
modules = sorted({k.split("base_model.model.")[1].split(".lora_")[0] for k in keep})
cfg["target_modules"] = modules
json.dump(cfg, open(dst / "adapter_config.json", "w"), indent=2)
for extra in ("README.md", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "chat_template.jinja"):
    if (src / extra).exists():
        shutil.copy2(src / extra, dst / extra)
print(f"{src.name} -> {dst.name}: kept {len(keep)} of {len(tensors)} tensors "
      f"({len(modules)} language-model modules); dropped groups: {dropped}")
