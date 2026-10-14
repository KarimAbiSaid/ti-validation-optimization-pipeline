"""
mask_recipe_discovery.py — named "recipe books": {mask_name: how-to-build-it}
dicts. Saved from Mask Generation — every build action there (atlas+region,
or Combine) can also upsert its own spec into a book, using Mask
Generation's full real capability (any atlas including Allen, hemisphere,
combine) rather than a separate, weaker reimplementation. Consumed by
Config Generation — every mask_name field there (ROI, non-ROI, subgroup/
background constraints) only ever offers names that exist in whichever
book is currently selected, and a "Build missing masks" action builds
exactly the ones a given subject doesn't already have.

Two recipe kinds, matching Mask Generation's two build mechanisms:
  "atlas_region" — {atlas, label_ids, hemisphere}, built via
                    discovery.generate_masks().
  "combine"      — {sources: [mask_name, ...]}, built via
                    discovery.combine_masks() once every source already
                    exists for that subject — sources can themselves be
                    atlas_region OR combine entries in the SAME book;
                    build_missing_masks() resolves the dependency order.

Pure Python, no Dash import.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import discovery
from common import PROJECT_DIR

RECIPE_BOOKS_DIR = Path(__file__).resolve().parent / "mask_recipes"

# Matches what the folder's own README/.gitignore treat as "not a real
# book" — never shown in listings, never overwritable by Save.
_RESERVED_STEMS = {"_template"}

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-. ]+$")


def _path_for(name: str) -> Path:
    return RECIPE_BOOKS_DIR / f"{name}.json"


def validate_book_name(name: str) -> str | None:
    """None if name is a usable recipe-book filename stem, else an error
    string. Same permissive letters/digits/_/-/./space rule as presets."""
    name = (name or "").strip()
    if not name:
        return "enter a recipe book name"
    if name in _RESERVED_STEMS:
        return f'"{name}" is reserved (the template file) — pick another name'
    if not _SAFE_NAME_RE.match(name):
        return "use only letters, digits, spaces, _, -, ."
    return None


def list_recipe_books() -> list[dict]:
    """[{"name", "path", "created_at", "mask_names"}] — a summary per book
    for a picker dropdown, alphabetical. Unreadable files are skipped, not
    errors (same convention as every other directory listing here)."""
    RECIPE_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in RECIPE_BOOKS_DIR.glob("*.json"):
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
            "mask_names": sorted((data.get("masks") or {}).keys()),
        })
    out.sort(key=lambda r: r["name"])
    return out


def load_recipe_book(name: str) -> dict | None:
    """The full book dict ({"name","created_at","masks": {...}}), or None
    if it doesn't exist / can't be read."""
    path = _path_for(name)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_book(name: str, data: dict) -> dict:
    err = validate_book_name(name)
    if err:
        return {"success": False, "path": None, "error": err}

    path = _path_for(name)
    existing = load_recipe_book(name)
    created_at = (existing or {}).get("created_at") or datetime.now().isoformat(timespec="seconds")

    payload = dict(data)
    payload["name"] = name
    payload["created_at"] = created_at

    RECIPE_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        return {"success": False, "path": str(path), "error": str(e)}
    return {"success": True, "path": str(path), "error": None}


def upsert_mask_recipe(book_name: str, mask_name: str, recipe: dict) -> dict:
    """Creates the book if it doesn't exist yet, or adds/replaces ONE entry
    if it does — every other entry already in the book is left untouched.
    Returns {"success", "path", "error", "replaced"} — replaced is True if
    mask_name already had a (now overwritten) recipe in this book."""
    book = load_recipe_book(book_name) or {"masks": {}}
    masks = dict(book.get("masks") or {})
    replaced = mask_name in masks
    masks[mask_name] = recipe
    book = dict(book)
    book["masks"] = masks
    result = _save_book(book_name, book)
    result["replaced"] = replaced
    return result


def delete_recipe_book(name: str) -> dict:
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


# ═════════════════════════════════════════════════════════════════════════════
# Build-from-recipe
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_build_order(masks: dict, mask_name: str, _seen: frozenset = frozenset()) -> list[str]:
    """[*every source mask_name transitively depends on (deepest first)...,
    mask_name] — a combine recipe's sources must exist before it can be
    built, and a source can itself be a combine of further sources.
    Raises ValueError on an unknown TOP-LEVEL name or a circular
    dependency. A combine source with no recipe in this book is treated as
    a leaf — e.g. a mask built before recipe books existed, or by some
    other means entirely — and simply isn't resolved further here; whether
    it actually exists on disk is checked directly in
    build_missing_masks()."""
    if mask_name in _seen:
        raise ValueError(f"circular recipe dependency involving '{mask_name}'")
    if mask_name not in masks:
        raise ValueError(f"'{mask_name}' is not in this recipe book")

    order = []
    recipe = masks[mask_name]
    if recipe.get("kind") == "combine":
        for src in recipe.get("sources") or []:
            if src in masks:
                order.extend(_resolve_build_order(masks, src, _seen | {mask_name}))
    order.append(mask_name)

    seen_names, deduped = set(), []
    for n in order:
        if n not in seen_names:
            seen_names.add(n)
            deduped.append(n)
    return deduped


def build_missing_masks(subject_ids: list[str], masks: dict, mask_names: list[str],
                        project_dir: str = PROJECT_DIR) -> list[dict]:
    """For each subject independently, builds whichever of mask_names (and
    anything a "combine" recipe among them transitively depends on) don't
    already exist for that subject. A mask already on disk for a given
    subject is never rebuilt, and a subject that already has everything it
    needs is never touched — this is the one real implementation both
    Config Generation's own "Build missing masks" button and Apply
    Preference's one-shot chain call, so there is exactly one place that
    ever writes a mask file from a recipe.

    masks: the book's "masks" dict (or an equivalent one assembled from a
    Preference's own frozen recipe copies — see one_shot_discovery.py).
    Returns one row per (subject, mask) actually attempted — nothing is
    returned for a mask a subject already had.
    {"subject_id", "mask_name", "success", "voxel_count", "error"}."""
    out = []
    for sid in subject_ids:
        for name in mask_names:
            try:
                order = _resolve_build_order(masks, name)
            except ValueError as e:
                out.append({"subject_id": sid, "mask_name": name, "success": False,
                           "voxel_count": None, "error": str(e)})
                continue
            for dep_name in order:
                path = discovery.expected_mask_path(sid, dep_name, project_dir)
                if os.path.isfile(path):
                    continue  # already there for this subject — never rebuilt

                recipe = masks[dep_name]
                if recipe.get("kind") == "combine":
                    src_paths = [discovery.expected_mask_path(sid, s, project_dir)
                                for s in recipe.get("sources") or []]
                    missing_src = [p for p in src_paths if not os.path.isfile(p)]
                    if missing_src:
                        out.append({"subject_id": sid, "mask_name": dep_name, "success": False,
                                   "voxel_count": None,
                                   "error": f"source mask(s) still missing: {missing_src}"})
                        continue
                    r = discovery.combine_masks(sid, src_paths, dep_name, project_dir)
                    out.append({"subject_id": sid, "mask_name": dep_name, "success": r["success"],
                               "voxel_count": r.get("voxel_count"), "error": r.get("error")})
                else:
                    r = discovery.generate_masks(
                        [sid], recipe["atlas"], recipe.get("label_ids") or {}, dep_name,
                        hemisphere=recipe.get("hemisphere"), project_dir=project_dir)[0]
                    out.append({"subject_id": sid, "mask_name": dep_name, "success": r["success"],
                               "voxel_count": r.get("voxel_count"), "error": r.get("error")})
    return out
