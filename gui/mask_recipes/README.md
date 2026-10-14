# Mask recipe books

This folder is git-tracked (so it exists for everyone who clones the repo),
but the actual `.json` recipe books saved into it are **not** — see
`.gitignore` in `code/` (the repo root — this project's `.git` lives in
`code/`, not at the top of `BIDS_TI_Toolbox/`). Each book is personal/local
by default.

**Sharing one:** there's no automatic sync. To share a book with a
colleague, either send them the `.json` file directly, or deliberately
force-add it past the ignore rule (run from inside `code/`):

```
git add -f gui/mask_recipes/<name>.json
git commit -m "Share <name> recipe book"
```

## What a recipe book is

A named `{mask_name: how-to-build-it}` dict. Saved from the Mask
Generation page — building a mask there (atlas+region, or Combine) has an
"also save this as a recipe" option, so a book is built up the same way you'd
build masks normally, using Mask Generation's full real capability (any
atlas including Allen, hemisphere, combine of existing masks) rather than a
separate, weaker reimplementation elsewhere.

Consumed by the Config Generation page: pick one recipe book, and every
mask_name field there (ROI, non-ROI, subgroup/background constraints) only
offers names that exist in it. For the subject(s) you have selected, a name
that already has a file on disk is just used; one that doesn't is flagged
"not built yet" and a "Build missing masks" button builds exactly the ones
missing for exactly the subjects missing them — a subject that already has
everything it needs is never touched, and an existing mask is never
rebuilt/overwritten by this.

Two recipe kinds:
- `"atlas_region"` — `{atlas, label_ids, hemisphere}`, built the same way
  Mask Generation's own atlas+region flow does.
- `"combine"` — `{sources: [mask_name, ...]}`, built the same way Mask
  Generation's own Combine flow does, once every source already exists for
  that subject. A source can itself be another entry in the same book
  (including another combine) — building resolves the dependency order
  automatically.

See `_template.json` in this folder for the exact shape.
