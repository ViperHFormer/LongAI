from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longai.utils.io import read_jsonl, write_json


@dataclass
class WriterSFTConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def _safe_subset(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: max(0, limit)]


def _resolve_lora_target_modules(model: Any) -> list[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = [name for name, _ in model.named_modules()]
    found = [m for m in preferred if any(n.endswith(m) for n in names)]
    if found:
        return found

    fallback = ["c_attn", "c_proj", "fc_in", "fc_out"]
    found_fb = [m for m in fallback if any(n.endswith(m) for n in names)]
    if found_fb:
        return found_fb

    # Last resort: use common linear suffixes to avoid hard failure on uncommon backbones.
    candidates = set()
    for name in names:
        tail = name.split(".")[-1]
        if tail in {"proj", "dense", "linear"}:
            candidates.add(tail)
    return sorted(candidates) if candidates else ["c_attn"]


def run_writer_sft(
    train_path: Path,
    output_dir: Path,
    eval_path: Path | None = None,
    config: WriterSFTConfig | None = None,
) -> dict:
    config = config or WriterSFTConfig()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        summary = {
            "status": "error",
            "message": "Missing training deps. Install trl, peft, datasets, bitsandbytes if using 4-bit.",
            "error": str(exc),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "writer_sft_summary.json", summary)
        return summary

    train_rows = _safe_subset(read_jsonl(train_path), config.max_train_samples)
    eval_rows: list[dict[str, Any]] = []
    if eval_path and eval_path.exists():
        eval_rows = _safe_subset(read_jsonl(eval_path), config.max_eval_samples)

    if not train_rows:
        summary = {"status": "error", "message": f"No training rows found in {train_path}"}
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "writer_sft_summary.json", summary)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    except Exception as exc:
        summary = {
            "status": "error",
            "message": "Failed to load model. Prefer a safetensors-based model or upgrade torch>=2.6.",
            "model_name": config.model_name,
            "error": str(exc),
        }
        write_json(output_dir / "writer_sft_summary.json", summary)
        return summary
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = _resolve_lora_target_modules(model)
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    trainer_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset is not None else "no",
        bf16=True,
        report_to=[],
        max_length=config.max_seq_length,
        dataset_text_field="text",
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))
    tokenizer.save_pretrained(str(output_dir / "checkpoint-final"))

    summary = {
        "status": "ok",
        "message": "Writer SFT completed.",
        "model_name": config.model_name,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "global_step": int(train_result.global_step),
        "train_loss": float(train_result.training_loss) if train_result.training_loss is not None else None,
        "checkpoint": str(output_dir / "checkpoint-final"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "writer_sft_summary.json", summary)
    return summary
