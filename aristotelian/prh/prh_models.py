"""PRH model registry and loaders."""

from __future__ import annotations

from typing import List, Tuple

import timm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_models(modelset: str, modality: str = "all") -> Tuple[List[str], List[str]]:
    """Return the exact model sets used in the paper."""
    if modality not in {"all", "vision", "language"}:
        raise ValueError(f"Unknown modality: {modality}")

    if modelset == "val":
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
        ]
        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "vit_base_patch16_224.augreg_in21k",
            "vit_large_patch16_224.augreg_in21k",
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
            "vit_huge_patch14_clip_224.laion2b",
            "vit_base_patch16_clip_224.laion2b_ft_in12k",
            "vit_large_patch14_clip_224.laion2b_ft_in12k",
            "vit_huge_patch14_clip_224.laion2b_ft_in12k",
        ]
    elif modelset == "videoprh":
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
            "google/gemma-2-9b-it",
        ]
        lvm_models = []
    else:
        raise ValueError(f"Unknown paper modelset: {modelset}")

    if modality == "vision":
        llm_models = []
    elif modality == "language":
        lvm_models = []
    return llm_models, lvm_models


def load_text_model(model_name: str, *, device: str = "cpu"):
    torch_dtype = None
    if device != "cpu" and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if "huggyllama" in model_name:
        tokenizer.pad_token = "[PAD]"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch_dtype is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, output_hidden_states=True, torch_dtype=torch_dtype
        )
    model.to(device)
    model.eval()
    return tokenizer, model


def load_vision_model(model_name: str, *, device: str = "cpu"):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()
    return model
