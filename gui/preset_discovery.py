"""
preset_discovery.py — list/save/load named Config Generation "presets":
everything on that page except which subject(s) it's applied to — cap,
ROI/non-ROI/every subgroup-constraint mask (by name), every optimizer/
electrode/flags setting — plus a FROZEN COPY of each referenced mask's
build recipe (atlas/label_ids/hemisphere, or a "combine" of other
recipes), copied out of whichever mask_recipes/*.json book they came from
at save time so a preset keeps behaving the same way even if that book is
later edited or deleted. See presets/README.md and presets/_template.json
for the exact shape, and mask_recipe_discovery.py for how those frozen
recipes actually get built — this module doesn't interpret the content,
just reads/writes it as JSON.

Pure Python, no Dash import.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent / "presets"

# Matches what the folder's own README/.gitignore treat as "not a real
# preset" — never shown in listings, never overwritable by Save.
_RESERVED_STEMS = {"_template"}

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-. ]+$")


def _path_for(name: str) -> Path:
    return PRESETS_DIR / f"{name}.json"


def validate_name(name: str) -> str | None:
    """None if name is a usable preset filename stem, else an error
    string. Deliberately permissive (letters/digits/_/-/./space) rather
    than trying to guess every OS's exact reserved-character set — this
    runs on Windows locally, so backslash/slash/colon etc. are rejected,
    which also happens to cover the other platforms' stricter sets."""
    name = (name or "").strip()
    if not name:
        return "enter a name"
    if name in _RESERVED_STEMS:
        return f'"{name}" is reserved (the template file) — pick another name'
    if not _SAFE_NAME_RE.match(name):
        return "use only letters, digits, spaces, _, -, ."
    return None


def list_presets() -> list[dict]:
    """[{"name", "path", "created_at", "cap_path", "roi_name", "non_roi_name"}]
    — a summary per preset for a picker dropdown/table, newest first.
    Unreadable files are skipped, not errors (same "don't block on one bad
    file" convention as everything else that lists a directory here)."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in PRESETS_DIR.glob("*.json"):
        if p.stem in _RESERVED_STEMS:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({
            "name": p.stem,
            "path": str(p),
            "created_at": data.get("created_at", ""),
            "cap_path": data.get("cap_path", ""),
            "roi_name": data.get("roi_name", ""),
            "non_roi_name": data.get("non_roi_name", ""),
        })
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out


def load_preset(name: str) -> dict | None:
    """The full preset dict, or None if it doesn't exist / can't be read."""
    path = _path_for(name)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_preset(name: str, data: dict, overwrite: bool = False) -> dict:
    """Writes one preset JSON. Returns {"success", "path", "error"}. Stamps
    "name"/"created_at" into the saved data itself (so a loaded preset is
    self-describing even if the file gets renamed) — created_at is only
    set on first save, preserved on overwrite, so "newest first" in
    list_presets() reflects when it was first created, not last edited."""
    err = validate_name(name)
    if err:
        return {"success": False, "path": None, "error": err}

    path = _path_for(name)
    if path.is_file() and not overwrite:
        return {"success": False, "path": str(path),
                "error": f'"{name}" already exists — confirm overwrite to replace it'}

    existing = load_preset(name) if path.is_file() else None
    created_at = (existing or {}).get("created_at") or datetime.now().isoformat(timespec="seconds")

    payload = dict(data)
    payload["name"] = name
    payload["created_at"] = created_at

    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        return {"success": False, "path": str(path), "error": str(e)}
    return {"success": True, "path": str(path), "error": None}


def delete_preset(name: str) -> dict:
    """Returns {"success", "error"}. Refuses to touch the reserved template."""
    if name in _RESERVED_STEMS:
        return {"success": False, "error": f'"{name}" is reserved (the template file)'}
    path = _path_for(name)
    if not path.is_file():
        return {"success": False, "error": f'"{name}" not found'}
    try:
        os.remove(path)
    except OSError as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "error": None}
