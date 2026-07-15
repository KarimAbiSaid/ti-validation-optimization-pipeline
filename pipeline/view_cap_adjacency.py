"""
view_cap_adjacency.py — visualise the convex-hull electrode adjacency graph.

Projects all cap electrodes onto a 2-D top-down head view (X = left/right,
Y = posterior/anterior) and draws the adjacency edges so you can visually
confirm which pairs would be excluded by the no_adjacent_electrodes constraint.

Usage:
    python view_cap_adjacency.py
    python view_cap_adjacency.py --highlight Cz P4
    python view_cap_adjacency.py --cap path/to/other_cap.csv
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAP_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "caps", "BioSemi32_MNE.csv"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cap", default=CAP_DEFAULT, help="Path to cap CSV")
    p.add_argument("--highlight", nargs="*", default=[],
                   help="Electrode(s) to highlight and list neighbours of")
    p.add_argument("--output", default=None, help="Output PNG path (default: same dir as cap)")
    return p.parse_args()


def load_cap(csv_path):
    cap_pos = {}
    with open(csv_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    if not lines:
        sys.exit("ERROR: cap CSV is empty")

    hdr = [c.strip().lower() for c in lines[0].replace(",", " ").split()]
    xi = next((i for i, h in enumerate(hdr) if h == "x"), None)
    yi = next((i for i, h in enumerate(hdr) if h == "y"), None)
    zi = next((i for i, h in enumerate(hdr) if h == "z"), None)
    ni = next((i for i, h in enumerate(hdr) if h in ("label","name","ch_name","channel")), None)
    has_header = xi is not None and yi is not None and zi is not None

    for line in (lines[1:] if has_header else lines):
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        try:
            if has_header:
                x, y, z = float(parts[xi]), float(parts[yi]), float(parts[zi])
                name = parts[ni] if ni is not None else parts[0]
            else:
                name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            cap_pos[name] = np.array([x, y, z])
        except (ValueError, IndexError):
            continue
    return cap_pos


def build_adjacency(cap_pos, cap_csv_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from run_pipeline import _BIOSEMI32_ADJACENCY

    names  = list(cap_pos.keys())
    pos3d  = np.array([cap_pos[n] for n in names])
    cap_name = os.path.splitext(os.path.basename(cap_csv_path))[0].lower()
    adj = set()

    if 'biosemi32' in cap_name:
        elec_set = set(names)
        for elec, nbs in _BIOSEMI32_ADJACENCY.items():
            if elec in elec_set:
                for nb in nbs:
                    if nb in elec_set:
                        adj.add(frozenset([elec, nb]))
        print(f"Using hardcoded BioSemi32 topology: {len(adj)} adjacent pairs")
    else:
        from scipy.spatial import ConvexHull, Delaunay
        for simplex in ConvexHull(pos3d).simplices:
            for a, b in combinations(simplex.tolist(), 2):
                adj.add(frozenset([names[a], names[b]]))
        for simplex in Delaunay(pos3d[:, :2]).simplices:
            for a, b in combinations(simplex.tolist(), 2):
                adj.add(frozenset([names[a], names[b]]))

    return names, pos3d, adj


def main():
    args = parse_args()
    cap_pos = load_cap(args.cap)
    names, pos, adj = build_adjacency(cap_pos, args.cap)

    # 2-D top-down projection: use X (left/right) and Y (posterior/anterior)
    x2d = pos[:, 0]
    y2d = pos[:, 1]
    name_to_idx = {n: i for i, n in enumerate(names)}

    highlight = [e for e in args.highlight if e in name_to_idx]
    for e in args.highlight:
        if e not in name_to_idx:
            print(f"WARNING: '{e}' not found in cap — skipping")

    _, ax = plt.subplots(figsize=(10, 10), facecolor="white")

    # Draw adjacency edges
    for pair in adj:
        a, b = list(pair)
        ia, ib = name_to_idx[a], name_to_idx[b]
        # Highlight edges connected to highlighted electrodes
        if a in highlight or b in highlight:
            ax.plot([x2d[ia], x2d[ib]], [y2d[ia], y2d[ib]],
                    color="tomato", lw=2.0, zorder=2)
        else:
            ax.plot([x2d[ia], x2d[ib]], [y2d[ia], y2d[ib]],
                    color="#aaaaaa", lw=0.8, zorder=1)

    # Draw electrodes
    for i, name in enumerate(names):
        is_hl = name in highlight
        color  = "tomato"  if is_hl else "steelblue"
        zorder = 5         if is_hl else 3
        size   = 120       if is_hl else 60
        ax.scatter(x2d[i], y2d[i], c=color, s=size, zorder=zorder, edgecolors="white", lw=0.8)
        ax.text(x2d[i], y2d[i] + 3, name, ha="center", va="bottom",
                fontsize=7.5, zorder=6, color="black",
                fontweight="bold" if is_hl else "normal")

    # Draw a rough head outline
    theta = np.linspace(0, 2*np.pi, 300)
    r = max(np.hypot(x2d, y2d)) * 1.12
    ax.plot(r * np.cos(theta), r * np.sin(theta), color="#cccccc", lw=1.2, zorder=0)
    # Nose indicator
    nose_x = [r*0.08, 0, -r*0.08]
    nose_y = [r*0.97, r*1.08, r*0.97]
    ax.plot(nose_x, nose_y, color="#cccccc", lw=1.2, zorder=0)

    cap_name = os.path.splitext(os.path.basename(args.cap))[0]
    n_elec = len(names)
    n_adj  = len(adj)
    avg_nb = round(n_adj * 2 / n_elec, 1)
    ax.set_title(
        f"{cap_name}  |  {n_elec} electrodes  |  {n_adj} adjacent pairs  "
        f"|  avg {avg_nb} neighbours/electrode\n"
        f"(top-down view, nose at top, left=left)",
        fontsize=10
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Print neighbour lists for highlighted electrodes
    if highlight:
        print("\nNeighbour lists:")
        for e in highlight:
            nb = sorted([
                list(p)[1] if list(p)[0] == e else list(p)[0]
                for p in adj if e in p
            ])
            print(f"  {e}: {nb}")

    out = args.output or os.path.join(
        os.path.dirname(args.cap),
        f"{cap_name}_adjacency.png"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
