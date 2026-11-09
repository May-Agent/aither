"""Deterministic tar.zst packaging of the compiled model and its metadata.

Entries are sorted by name and stamped with fixed mtime/uid/gid/mode, so two runs over
the same compiled model produce identical bytes. The archive holds the compiled model
and its `manifest.json` (the shape `ondevice-coreml` loads); the tokenizer is pinned
separately, at the ONNX repository's revision, because the ONNX build shares it.
"""

import io
import json
import tarfile
from pathlib import Path

import zstandard

from .fetch import sha256_file
from .spec import BUCKETS, DEFAULT_FUNCTION, FUNCTION_PREFIX, SHORT_REVISION, SOURCE_REPO, SOURCE_REVISION

NOTICE = f"""Language detection model, by Alexis Neakameni.
Source: https://huggingface.co/{SOURCE_REPO}
Pinned revision: {SOURCE_REVISION}
Licensed under the MIT License.
Converted to Core ML (float16) for aither.
"""


def archive_manifest(labels: int) -> dict:
    return {
        "model": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "functions": {f"{FUNCTION_PREFIX}{b}": b for b in BUCKETS},
        "default_function": DEFAULT_FUNCTION,
        "inputs": {
            "input_ids": {"dtype": "int32", "shape": [1, "length"]},
            "attention_mask": {"dtype": "float16", "shape": [1, "length"]},
        },
        "outputs": {"logits": {"dtype": "float16", "shape": [1, labels]}},
    }


def _info(name: str, size: int, dir_: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if dir_:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.mode = 0o644
        info.size = size
    return info


def _iter_mlmodelc(mlmodelc: Path) -> list[tuple[str, Path | None]]:
    """(archive name, fs path or None for a directory) entries."""
    entries = [(mlmodelc.name + "/", None)]
    for p in sorted(mlmodelc.rglob("*")):
        rel = p.relative_to(mlmodelc.parent)
        if p.is_dir():
            entries.append((f"{rel}/", None))
        elif p.is_file():
            entries.append((str(rel), p))
        else:
            raise RuntimeError(f"unsupported entry in mlmodelc (not a file or directory): {p}")
    return entries


def build_archive(work_dir: Path, mlmodelc: Path, labels: int) -> Path:
    out = work_dir / f"language-coreml-{SHORT_REVISION}.tar.zst"
    entries: list[tuple[str, object]] = _iter_mlmodelc(mlmodelc)
    entries.append(("NOTICE", NOTICE.encode()))
    entries.append(("manifest.json", (json.dumps(archive_manifest(labels), indent=2) + "\n").encode()))
    entries.sort(key=lambda e: e[0])
    with out.open("wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw) as zstd:
            with tarfile.open(fileobj=zstd, mode="w|") as tar:
                for name, payload in entries:
                    if payload is None:
                        tar.addfile(_info(name, 0, dir_=True))
                    elif isinstance(payload, bytes):
                        tar.addfile(_info(name, len(payload)), io.BytesIO(payload))
                    else:
                        with Path(payload).open("rb") as f:
                            tar.addfile(_info(name, Path(payload).stat().st_size), f)
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"wrote {out} size={out.stat().st_size} sha256={digest}", flush=True)
    return out
