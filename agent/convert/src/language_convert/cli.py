"""Pipeline entry point: fetch, convert, verify, package."""

import argparse
import gc
from pathlib import Path

import torch
from transformers import BertForSequenceClassification

from .convert import compile_plan_variants, convert_all
from .fetch import fetch_flores, fetch_sources
from .package import build_archive
from .verify import verify


def main() -> None:
    parser = argparse.ArgumentParser(prog="aither-language-convert")
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    sources = fetch_sources(work_dir)
    sentences = fetch_flores(work_dir)
    weights = sources["alexneakameni/language_detection/model.safetensors"].parent
    model = BertForSequenceClassification.from_pretrained(
        weights, attn_implementation="eager", dtype=torch.float32
    ).eval()

    mlpackage, mlmodelc, _ = convert_all(model, build_dir)
    plan_variants = compile_plan_variants(mlpackage, build_dir)
    verify(
        model,
        mlmodelc,
        plan_variants,
        sources["onnx-community/language_detection-ONNX/tokenizer.json"],
        sources["onnx-community/language_detection-ONNX/onnx/model_int8.onnx"],
        sentences,
        work_dir,
    )
    build_archive(work_dir, mlmodelc, model.config.num_labels)
    gc.collect()
