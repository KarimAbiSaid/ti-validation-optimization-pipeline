#!/usr/bin/env python3
"""
generate_biosemi32_cap.py

Extracts BioSemi32 electrode positions from SimNIBS's built-in
EEG10-10_UI_Jurak_2007 cap (MNI152 mm space) and writes them to
configs/caps/BioSemi32_MNE.csv in SimNIBS cap format.

Usage (local, SimNIBS installed):
    python generate_biosemi32_cap.py

Output:
    configs/caps/BioSemi32_MNE.csv

Coordinate system
-----------------
SimNIBS built-in caps (ElectrodeCaps_MNI/) use MNI152 space (mm).
We pull directly from that validated source — no coordinate conversion,
no Procrustes approximation.  register_caps.py then uses
mni2subject_coords() to warp these exact MNI positions to each subject.
"""

import csv
import sys
import simnibs
from pathlib import Path

# ---------------------------------------------------------------------------
# BioSemi32 electrode order (must stay fixed — leadfield index depends on it)
# ---------------------------------------------------------------------------
BIOSEMI32_ORDER = [
    "Fp1", "AF3", "F7",  "F3",  "FC1", "FC5", "T7",  "C3",
    "CP1", "CP5", "P7",  "P3",  "Pz",  "PO3", "O1",  "Oz",
    "O2",  "PO4", "P4",  "P8",  "CP6", "CP2", "C4",  "T8",
    "FC6", "FC2", "F4",  "F8",  "AF4", "Fp2", "Fz",
]
REFERENCE = "Cz"

# ---------------------------------------------------------------------------
# Load MNI positions from SimNIBS built-in 10-10 cap
# ---------------------------------------------------------------------------
simnibs_dir = Path(simnibs.__file__).parent
jurak_csv   = simnibs_dir / "resources" / "ElectrodeCaps_MNI" / "EEG10-10_UI_Jurak_2007.csv"

if not jurak_csv.exists():
    sys.exit(f"ERROR: could not find SimNIBS Jurak 2007 cap at:\n  {jurak_csv}")

mni_positions = {}
with open(jurak_csv, newline="") as f:
    for row in csv.reader(f):
        if not row or row[0] not in ("Electrode", "ReferenceElectrode"):
            continue
        label = row[4].strip()
        mni_positions[label] = (float(row[1]), float(row[2]), float(row[3]))

# Verify all BioSemi32 electrodes are present
missing = [e for e in BIOSEMI32_ORDER + [REFERENCE] if e not in mni_positions]
if missing:
    sys.exit(f"ERROR: missing from Jurak 2007 cap: {missing}")

print(f"Source: {jurak_csv.name}  ({len(mni_positions)} electrodes total)")

# ---------------------------------------------------------------------------
# Write output CSV
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent / "configs" / "caps" / "BioSemi32_MNE.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Type", "x", "y", "z", "label"])
    for name in BIOSEMI32_ORDER:
        x, y, z = mni_positions[name]
        w.writerow(["Electrode", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", name])
    x, y, z = mni_positions[REFERENCE]
    w.writerow(["ReferenceElectrode", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", REFERENCE])

print(f"Saved  → {out_path}")
print(f"  {len(BIOSEMI32_ORDER)} electrodes + {REFERENCE} (reference)")
print()
print("Spot-check MNI positions (compare to old Procrustes values):")
checks = {
    "Fp1": "was (-29.3,  99.9,  29.7)",
    "T7":  "was (-94.9,  11.1,  13.4)",
    "FC1": "was (-35.6,  30.9, 102.3)",
    "FC2": "was ( 35.6,  30.9, 102.3)",
    "T8":  "was ( 94.9,  11.1,  13.4)",
    "Cz":  "was (  0.0,  -6.7, 110.1)",
}
for label, note in checks.items():
    x, y, z = mni_positions[label]
    print(f"  {label:6s}: ({x:7.2f}, {y:7.2f}, {z:7.2f})   {note}")
