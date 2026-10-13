"""Artifact integrity and portable paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def verify_manifest(root, manifest):
    entries = read_json(manifest)["files"]
    seen = set()
    for entry in entries:
        name = entry["path"]
        if name in seen:
            raise ValueError(f"Duplicate artifact: {name}")
        seen.add(name)
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact integrity check failed: {name}")
    return {"verified_files": len(entries), "bytes": sum(r["bytes"] for r in entries)}

