"""Verification of the converted model.

Three checks, any miss fails the run:

- the traced module computes the transformers model's logits (fp32, near-exact);
- the Core ML model picks the same language as the fp32 weights and as the int8
  ONNX model the Linux and Windows builds run, on FLORES-200 sentences, on at least
  MIN_AGREEMENT of them (the guard needs the same top label, not equal logits);
- at least MIN_ANE_SHARE of each function's operations are placed on the Neural
  Engine.

MLComputePlan reports compute-device usage only for the compiled model's default
function, so each function is counted on a variant compiled with it as default.
Arrays handed to ``predict`` are numpy-owned copies: a numpy view of a torch tensor
makes Core ML free torch memory on a thread without the GIL.
"""

import gc
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort
import torch
from coremltools.models.compute_plan import MLComputePlan
from tokenizers import Tokenizer

from .encoder import Classifier
from .spec import BUCKETS, FUNCTION_PREFIX, MIN_AGREEMENT, MIN_ANE_SHARE, SHORT_REVISION


def _device_name(usage) -> str:
    return type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")


def _op_counts(plan, function: str) -> dict:
    ops = plan.model_structure.program.functions[function].block.operations
    counts = {}
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            if op.operator_name == "const":
                continue
            raise RuntimeError(f"no compute device usage for non-const op {op.operator_name}")
        name = _device_name(usage)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _bucket(length: int) -> int:
    for b in BUCKETS:
        if length <= b:
            return b
    raise RuntimeError(f"{length} tokens exceed the largest bucket {BUCKETS[-1]}")


def verify(
    model,
    mlmodelc: Path,
    plan_variants: dict[int, Path],
    tokenizer_path: Path,
    onnx_path: Path,
    sentences: dict[str, list[str]],
    work_dir: Path,
) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=BUCKETS[-1])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = [i.name for i in session.get_inputs()]
    failures = []
    report = {
        "source": {"repo_id": "alexneakameni/language_detection", "revision_prefix": SHORT_REVISION},
        "checks": {"min_agreement": MIN_AGREEMENT, "min_ane_share": MIN_ANE_SHARE},
        "ane": {},
    }

    for b in BUCKETS:
        fn = f"{FUNCTION_PREFIX}{b}"
        plan = MLComputePlan.load_from_path(
            path=str(plan_variants[b]), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        counts = _op_counts(plan, fn)
        del plan
        gc.collect()
        total = sum(counts.values())
        share = counts.get("NeuralEngine", 0) / total
        print(f"{fn}: {counts} ({share:.1%} on the Neural Engine)", flush=True)
        report["ane"][fn] = {"ops": counts, "ane_share": share}
        if share < MIN_ANE_SHARE:
            failures.append(f"{fn}: {share:.1%} of operations on the Neural Engine < {MIN_ANE_SHARE:.0%}")

    classifiers = {b: Classifier(model, b).eval() for b in BUCKETS}
    compiled = {
        b: ct.models.CompiledMLModel(
            str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=f"{FUNCTION_PREFIX}{b}"
        )
        for b in BUCKETS
    }
    total = 0
    same_torch = same_onnx = onnx_vs_torch = 0
    worst_port = 0.0
    per_language = {}
    keepalive = []
    for language, texts in sentences.items():
        agree = 0
        for text in texts:
            encoding = tokenizer.encode(text)
            ids = np.array(encoding.ids, dtype=np.int64)
            n = len(ids)
            b = _bucket(n)

            with torch.no_grad():
                ref = model(
                    input_ids=torch.tensor(ids).unsqueeze(0),
                    attention_mask=torch.ones(1, n, dtype=torch.long),
                ).logits[0].numpy()
                padded_ids = torch.zeros(1, b, dtype=torch.int32)
                padded_ids[0, :n] = torch.tensor(ids, dtype=torch.int32)
                padded_mask = torch.zeros(1, b)
                padded_mask[0, :n] = 1.0
                port = classifiers[b](padded_ids, padded_mask)[0].numpy()
            worst_port = max(worst_port, float(np.abs(port - ref).max()))

            feed = {
                "input_ids": np.array(padded_ids.numpy(), dtype=np.int32),
                "attention_mask": np.array(padded_mask.numpy(), dtype=np.float16),
            }
            keepalive.append(feed)
            coreml = compiled[b].predict(feed)["logits"][0]

            onnx_feed = {
                "input_ids": ids[None],
                "attention_mask": np.ones((1, n), dtype=np.int64),
                "token_type_ids": np.zeros((1, n), dtype=np.int64),
            }
            onnx = session.run(None, {k: onnx_feed[k] for k in onnx_inputs})[0][0]

            total += 1
            same_torch += int(np.argmax(coreml) == np.argmax(ref))
            same_onnx += int(np.argmax(coreml) == np.argmax(onnx))
            onnx_vs_torch += int(np.argmax(onnx) == np.argmax(ref))
            agree += int(np.argmax(coreml) == np.argmax(onnx))
        per_language[language] = agree / len(texts)
    del classifiers, compiled, keepalive
    gc.collect()

    report["sentences"] = total
    report["worst_port_logit_difference_fp32"] = worst_port
    report["coreml_vs_fp32_agreement"] = same_torch / total
    report["coreml_vs_onnx_int8_agreement"] = same_onnx / total
    report["onnx_int8_vs_fp32_agreement"] = onnx_vs_torch / total
    report["coreml_vs_onnx_int8_by_language"] = per_language
    print(json.dumps({k: v for k, v in report.items() if k != "ane"}, indent=2), flush=True)

    if worst_port > 1e-3:
        failures.append(f"the traced module differs from transformers by {worst_port} in logits")
    if same_torch / total < MIN_AGREEMENT:
        failures.append(f"Core ML agrees with the fp32 weights on {same_torch / total:.3%} < {MIN_AGREEMENT:.0%}")
    if same_onnx / total < MIN_AGREEMENT:
        failures.append(f"Core ML agrees with the ONNX int8 model on {same_onnx / total:.3%} < {MIN_AGREEMENT:.0%}")

    report["passed"] = not failures
    report["failures"] = failures
    out_path = work_dir / "verification.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)
    if failures:
        raise RuntimeError("verification failed: " + "; ".join(failures))
    return report
