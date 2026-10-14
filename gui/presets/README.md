# Config Generation presets

This folder is git-tracked (so it exists for everyone who clones the repo),
but the actual preset `.json` files saved into it are **not** — see
`.gitignore` in `code/` (the repo root — this project's `.git` lives in
`code/`, not at the top of `BIDS_TI_Toolbox/`). Each preset is
personal/local by default.

**Sharing one:** there's no automatic sync. To share a preset with a
colleague, either send them the `.json` file directly, or deliberately
force-add it past the ignore rule (run from inside `code/`):

```
git add -f gui/presets/<name>.json
git commit -m "Share <name> preset"
```

## What a preset is

A named bundle of everything on the Config Generation page **except which
subject(s) it's applied to** — cap, which mask is ROI/non-ROI, every
subgroup/background constraint mask, every optimizer/electrode/flags
setting — plus a **frozen copy** of every one of those masks' build
recipes, copied out of whichever `mask_recipes/*.json` book they were
picked from at save time (see `../mask_recipes/README.md`). Frozen on
purpose: a preset keeps behaving the same way later even if that book is
edited or deleted afterward.

Saved from the Config Generation page's "Save as Preference" button —
refused if any referenced mask has no recipe anywhere in the selected
book, since that would make the preset unbuildable on a new subject.
Applied from the Apply Preference page, which runs charm (if needed,
opt-in) → builds whichever masks a subject is still missing → cap
registration → config generation, for one or more subjects, in that
order. It never submits the actual optimization/simulation job — that
stays a separate, manual step.

See `_template.json` in this folder for the exact shape.
