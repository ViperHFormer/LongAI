from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_INSTANCE = None
_LLM_MODEL_NAME = None


def _get_llm(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    global _LLM_INSTANCE, _LLM_MODEL_NAME
    if _LLM_INSTANCE is not None and _LLM_MODEL_NAME == model_name:
        return _LLM_INSTANCE

    # Clear previous model to free GPU memory
    if _LLM_INSTANCE is not None:
        del _LLM_INSTANCE
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Prefer bf16 for faster inference; fall back to 8-bit if GPU memory is tight
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: 8-bit quantization
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

    _LLM_INSTANCE = (model, tokenizer)
    _LLM_MODEL_NAME = model_name
    return _LLM_INSTANCE


def reset_llm():
    """Release the cached LLM to free GPU memory."""
    global _LLM_INSTANCE, _LLM_MODEL_NAME
    if _LLM_INSTANCE is not None:
        del _LLM_INSTANCE
        _LLM_INSTANCE = None
        _LLM_MODEL_NAME = None
        torch.cuda.empty_cache()


def generate(prompt: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 1024) -> str:
    model, tokenizer = _get_llm(model_name)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
