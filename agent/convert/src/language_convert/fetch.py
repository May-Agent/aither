"""Fetch the pinned source files and verify them against the committed manifest.

LFS files carry a sha256 in Hub metadata; small files are plain git blobs and the
Hub publishes only their blob oid, so those are verified as git blob sha1
(sha1 of "blob <size>\\0" + contents). FLORES-200 is verified by the sha256 of its
archive.
"""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from .spec import FLORES_LANGUAGES, FLORES_LINES

_MANIFEST = Path(__file__).resolve().parents[2] / "manifest.json"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path, size: int) -> str:
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, name: str, meta: dict) -> None:
    size = path.stat().st_size
    if size != meta["size"]:
        raise RuntimeError(f"{name}: size {size} != manifest {meta['size']}")
    if "sha256" in meta:
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"{name}: sha256 {digest} != manifest {meta['sha256']}")
    elif "git_blob_sha1" in meta:
        digest = _git_blob_sha1(path, size)
        if digest != meta["git_blob_sha1"]:
            raise RuntimeError(
                f"{name}: git blob sha1 {digest} != manifest {meta['git_blob_sha1']}"
            )
    else:
        raise RuntimeError(f"{name}: manifest entry has no hash")
    print(f"fetched {name} ({size} bytes, hash ok)", flush=True)


def fetch_sources(work_dir: Path) -> dict[str, Path]:
    """Download every manifest file into work_dir/hub/<repo>, verified.

    Returns the local path of each file by `<repo_id>/<file>`.
    """
    manifest = json.loads(_MANIFEST.read_text())
    paths = {}
    for repo in manifest["hub"]:
        dest = work_dir / "hub" / repo["repo_id"].replace("/", "--")
        dest.mkdir(parents=True, exist_ok=True)
        for name, meta in repo["files"].items():
            path = Path(
                hf_hub_download(
                    repo_id=repo["repo_id"],
                    filename=name,
                    revision=repo["revision"],
                    local_dir=dest,
                )
            )
            _check(path, name, meta)
            paths[f"{repo['repo_id']}/{name}"] = path
    return paths


def fetch_flores(work_dir: Path) -> dict[str, list[str]]:
    """The first FLORES_LINES devtest sentences of each checked language."""
    meta = json.loads(_MANIFEST.read_text())["flores"]
    archive = work_dir / "flores200_dataset.tar.gz"
    if not archive.is_file():
        with urllib.request.urlopen(meta["url"]) as response, archive.open("wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
    _check(archive, "flores200_dataset.tar.gz", meta)
    sentences = {}
    with tarfile.open(archive) as tar:
        for language in FLORES_LANGUAGES:
            member = tar.extractfile(f"./flores200_dataset/devtest/{language}.devtest")
            if member is None:
                raise RuntimeError(f"FLORES-200 has no devtest file for {language}")
            lines = member.read().decode("utf-8").splitlines()
            sentences[language] = lines[:FLORES_LINES]
    return sentences
