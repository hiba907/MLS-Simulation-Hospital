"""Suture Lab — reference gallery + simulator for common suture patterns.

Provides:
- SUTURE_TYPES: dict of 8 suture patterns with descriptions, indications, and
  target stitch positions for the simulator.
- render_schematic(): draw a clean matplotlib schematic of any pattern.
- score_stitch(): score a student capture against a target stitch.

This module is pure-Python and has no Streamlit dependency so it can be
unit-tested. The surgery tab in mls_hospital_module.py uses these functions.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── 8 common suture patterns ────────────────────────────────────────────────
# Each entry:
#   name        : display name
#   short       : 1-line description
#   description : 2-3 line clinical description
#   indications : when to use it
#   n_bites     : number of "bites" (needle passes through tissue) per stitch
#   n_stitches  : number of stitches in the pattern
#   bites(width, height, n_stitches)
#               : function that returns list of (x, y) target positions in
#                 normalized [0..1] coordinates. The wound runs along x=0.5.
#                 (0, 0) is top-left of the practice frame, (1, 1) is
#                 bottom-right. Bites alternate above and below the wound line.

# Reusable: a basic interrupted pattern of n stitches at evenly spaced x
def _interrupted(width, height, n_stitches=4, offset=0.10, above=0.30, below=0.30):
    """A simple interrupted pattern: each stitch is two bites (above + below)
    at the same x. Stitches evenly spaced across the wound width."""
    bites = []
    margin = (1.0 - width) / 2
    for i in range(n_stitches):
        x = margin + width * (i + 0.5) / n_stitches
        bites.append((x, 0.5 - above))  # entry, above wound
        bites.append((x, 0.5 + below))  # exit, below wound
    return bites


def _continuous(width, height, n_stitches=4, above=0.30, below=0.30):
    """Simple continuous (running) suture: thread zig-zags across the wound
    without a knot between stitches."""
    bites = []
    margin = (1.0 - width) / 2
    side = -1  # start above
    for i in range(n_stitches):
        x = margin + width * (i + 0.5) / n_stitches
        side = -side
        y = 0.5 + side * (above if side < 0 else below)
        bites.append((x, y))
        # For continuous, the bite is in/out at adjacent x positions
        if i < n_stitches - 1:
            x_next = margin + width * (i + 1.5) / n_stitches
            y_other = 0.5 - side * (above if side < 0 else below)
            bites.append((x_next, y_other))
    return bites


def _continuous_locking(width, height, n_stitches=4, above=0.30, below=0.30):
    """Ford interlocking — like continuous but each pass loops through the
    previous loop on the opposite side."""
    # Same targets as continuous for the simulator; visual difference in diagram
    return _continuous(width, height, n_stitches, above, below)


def _figure_of_eight(width, height, n_stitches=3, above=0.30, below=0.30):
    """Figure-of-8 suture: each stitch crosses the wound in a figure-8 pattern.
    Two bites per stitch: one above, one below — same as interrupted for simulator
    purposes (the visual difference is the loop crossing)."""
    return _interrupted(width, height, n_stitches, 0.10, above, below)


def _vertical_mattress(width, height, n_stitches=3, far=0.45, near=0.15):
    """Vertical mattress (Donati): each stitch is 4 bites — far-far then
    near-near. Used for eversion of wound edges."""
    bites = []
    margin = (1.0 - width) / 2
    for i in range(n_stitches):
        x = margin + width * (i + 0.5) / n_stitches
        # far bite on near side
        bites.append((x, 0.5 - far))
        # far bite on far side
        bites.append((x, 0.5 + far))
        # near bite on far side
        bites.append((x, 0.5 + near))
        # near bite on near side
        bites.append((x, 0.5 - near))
    return bites


def _horizontal_mattress(width, height, n_stitches=3, far=0.20, near=0.20):
    """Horizontal mattress: each stitch is 4 bites — cross the wound at two
    x positions (near and far) on each side."""
    bites = []
    margin = (1.0 - width) / 2
    for i in range(n_stitches):
        x_center = margin + width * (i + 0.5) / n_stitches
        x_offset = 0.04
        # bite 1: above, near x
        bites.append((x_center - x_offset, 0.5 - near))
        # bite 2: above, far x
        bites.append((x_center + x_offset, 0.5 - far))
        # bite 3: below, far x
        bites.append((x_center + x_offset, 0.5 + far))
        # bite 4: below, near x
        bites.append((x_center - x_offset, 0.5 + near))
    return bites


def _subcuticular(width, height, n_stitches=4, depth=0.10):
    """Subcuticular (running subcutaneous): hidden bites just under the skin.
    The simulator targets are right at the wound edge (no large above/below)."""
    bites = []
    margin = (1.0 - width) / 2
    side = -1
    for i in range(n_stitches):
        x = margin + width * (i + 0.5) / n_stitches
        side = -side
        y = 0.5 + side * depth
        bites.append((x, y))
        if i < n_stitches - 1:
            x_next = margin + width * (i + 1.5) / n_stitches
            y_other = 0.5 - side * depth
            bites.append((x_next, y_other))
    return bites


def _purse_string(width, height, n_stitches=6, radius=0.30):
    """Purse-string: circular stitches around an opening. Targets are arranged
    on a circle in the middle of the frame."""
    cx, cy = 0.5, 0.5
    bites = []
    for i in range(n_stitches):
        ang = 2 * math.pi * i / n_stitches - math.pi / 2  # start at top
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        bites.append((x, y))
    return bites


SUTURE_TYPES = {
    "Simple Interrupted": {
        "name":        "Simple Interrupted",
        "short":       "Each stitch tied off separately — the workhorse closure",
        "description": ("Each stitch is a single pass through the tissue tied off "
                        "with its own knot. Most versatile pattern; used in most "
                        "skin closures, laparotomies, and routine closures."),
        "indications": "General skin closure, laparotomy, anywhere precise tension control is needed",
        "n_bites_per_stitch": 2,
        "n_stitches": 4,
        "icon":        "‖‖‖‖",
        "color":       "#1f77b4",
        "bites":       _interrupted,
    },
    "Simple Continuous (Running)": {
        "name":        "Simple Continuous (Running)",
        "short":       "One thread, multiple stitches, faster but less precise",
        "description": ("A single thread runs back-and-forth across the wound, "
                        "with knots only at the start and end. Faster than "
                        "interrupted but if it breaks, the whole line fails."),
        "indications": "Long incisions, vascular anastomoses, peritoneum",
        "n_bites_per_stitch": 1,
        "n_stitches": 4,
        "icon":        "∿∿∿",
        "color":       "#ff7f0e",
        "bites":       _continuous,
    },
    "Continuous Locking (Ford)": {
        "name":        "Continuous Locking (Ford Interlocking)",
        "short":       "Like continuous but each pass locks the previous",
        "description": ("Modified running suture where each stitch passes through "
                        "the loop of the previous one. Better for areas under "
                        "tension because each stitch is locked in place."),
        "indications": "Fascia closure, areas under tension",
        "n_bites_per_stitch": 1,
        "n_stitches": 4,
        "icon":        "⌇⌇⌇",
        "color":       "#2ca02c",
        "bites":       _continuous_locking,
    },
    "Figure-of-8": {
        "name":        "Figure-of-8",
        "short":       "Single stitch crosses the wound in an 8 pattern",
        "description": ("A single suture crosses the wound twice, forming an 8 "
                        "shape. Provides good hemostasis (stops bleeding) and is "
                        "self-locking. Often used for vessels or small defects."),
        "indications": "Hemostasis, small defects, vessel ligation",
        "n_bites_per_stitch": 2,
        "n_stitches": 3,
        "icon":        "∞∞",
        "color":       "#d62728",
        "bites":       _figure_of_eight,
    },
    "Vertical Mattress (Donati)": {
        "name":        "Vertical Mattress (Donati)",
        "short":       "Far-far-near-near — evens wound edges",
        "description": ("Each stitch has 4 bites: far-far first to take the deep "
                        "tension, then near-near to precisely evert the wound "
                        "edges. Best pattern for skin edge eversion."),
        "indications": "Skin closure under tension, everting edges (face, hands)",
        "n_bites_per_stitch": 4,
        "n_stitches": 3,
        "icon":        "⫶⫶⫶",
        "color":       "#9467bd",
        "bites":       _vertical_mattress,
    },
    "Horizontal Mattress": {
        "name":        "Horizontal Mattress",
        "short":       "Tension-bearing, everts edges strongly",
        "description": ("Each stitch is two parallel bites — one near the wound "
                        "and one further away on each side. High tension-bearing "
                        "capacity but can cause tissue strangulation if too tight."),
        "indications": "High-tension closures, tendon repairs",
        "n_bites_per_stitch": 4,
        "n_stitches": 3,
        "icon":        "≡≡",
        "color":       "#8c564b",
        "bites":       _horizontal_mattress,
    },
    "Subcuticular (Running Subcutaneous)": {
        "name":        "Subcuticular (Running Subcutaneous)",
        "short":       "Hidden stitch just under the skin — cosmetic",
        "description": ("A continuous stitch that runs just under the skin surface, "
                        "alternating sides. No visible knots, no cross-hatch scars. "
                        "Often used with absorbable material so removal isn't needed."),
        "indications": "Cosmetic closures (face, neck), pediatric closures",
        "n_bites_per_stitch": 1,
        "n_stitches": 4,
        "icon":        "∼∼",
        "color":       "#e377c2",
        "bites":       _subcuticular,
    },
    "Purse-string": {
        "name":        "Purse-string",
        "short":       "Circular stitch around an opening, cinches closed",
        "description": ("Stitches placed in a circle around an opening (appendiceal "
                        "stump, trocar site, stoma). Pulling the thread cinches the "
                        "opening closed like a drawstring bag."),
        "indications": "Appendiceal stump, trocar site closure, stoma creation",
        "n_bites_per_stitch": 1,
        "n_stitches": 6,
        "icon":        "⊙",
        "color":       "#17becf",
        "bites":       _purse_string,
    },
}


def render_schematic(suture_type: str, ax=None, show_targets: bool = True):
    """Draw a clean schematic of the chosen suture pattern on a matplotlib axis.
    Returns the figure. Used in the Suture Lab gallery.
    """
    info = SUTURE_TYPES[suture_type]
    color = info["color"]
    n_stitches = info["n_stitches"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    # Wound — a horizontal line in the middle of the frame
    if suture_type == "Purse-string":
        # Draw a circle as the "wound opening"
        ax.add_patch(plt.Circle((0.5, 0.5), 0.10, fill=False, edgecolor="#8b0000", linewidth=3))
        ax.text(0.5, 0.5, "Opening", ha="center", va="center", fontsize=8, color="#8b0000")
    else:
        ax.plot([0.05, 0.95], [0.5, 0.5], color="#8b0000", linewidth=3, solid_capstyle="butt")
        # Skin shading above and below
        ax.fill_between([0.05, 0.95], 0.5, 1.0, color="#fdebd0", alpha=0.5)
        ax.fill_between([0.05, 0.95], 0.0, 0.5, color="#fdebd0", alpha=0.5)

    # Get target points
    targets = info["bites"](0.80, 1.0)

    # Draw the stitches (line through targets in the order they should be made)
    if targets:
        xs = [p[0] for p in targets]
        ys = [p[1] for p in targets]
        # Stitches
        if suture_type == "Purse-string":
            # Connect in a circle
            for i in range(len(targets)):
                a, b = targets[i], targets[(i + 1) % len(targets)]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.4, alpha=0.75)
        else:
            # Connect in groups of n_bites_per_stitch
            n_bites = info["n_bites_per_stitch"]
            for s in range(n_stitches):
                start = s * n_bites
                end = start + n_bites
                bx = xs[start:end]
                by = ys[start:end]
                ax.plot(bx, by, color=color, linewidth=1.4, alpha=0.8, marker="o", markersize=4)

            # For continuous, draw the connecting thread between stitches
            if "Continuous" in suture_type or "Subcuticular" in suture_type:
                ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.5, linestyle="--")

    # Draw target dots
    if show_targets:
        for i, (x, y) in enumerate(targets):
            ax.plot(x, y, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.0)
            if i % info["n_bites_per_stitch"] == 0:
                ax.annotate(str(i // info["n_bites_per_stitch"] + 1),
                            (x, y), xytext=(-12, 12), textcoords="offset points",
                            fontsize=8, color=color, fontweight="bold")

    # Knot indicators
    if suture_type in ("Simple Interrupted", "Vertical Mattress (Donati)", "Horizontal Mattress", "Figure-of-8"):
        # Knot at the top of each stitch
        for s in range(n_stitches):
            x = 0.1 + 0.8 * (s + 0.5) / n_stitches
            ax.plot(x, 0.85, "*", color=color, markersize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(suture_type, fontsize=12, fontweight="bold", color=color, pad=10)

    return fig


def render_gallery_grid(types=None, cols=4):
    """Render a 4-column gallery of schematics. Returns the figure."""
    if types is None:
        types = list(SUTURE_TYPES.keys())
    rows = (len(types) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for i, name in enumerate(types):
        render_schematic(name, ax=axes[i], show_targets=True)
    # Hide unused axes
    for j in range(len(types), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def score_stitch(actual, target, frame_w: float = 1.0, max_dist: float = 0.18):
    """Score a single stitch capture.

    actual   : (x, y) in normalized [0..1] coordinates (camera frame)
    target   : (x, y) in normalized [0..1] coordinates (wound frame)
    frame_w  : width of the camera frame in normalized units (1.0 = full)
    max_dist : distance at which score reaches 0 (default 18% of frame)

    Returns (score: 0-100, distance: in normalized units, status: str)
    """
    dist = math.hypot(actual[0] - target[0], actual[1] - target[1])
    if dist >= max_dist:
        return 0.0, dist, "miss"
    score = 100.0 * (1.0 - dist / max_dist)
    if score >= 90:
        status = "excellent"
    elif score >= 75:
        status = "good"
    elif score >= 55:
        status = "fair"
    else:
        status = "poor"
    return score, dist, status


# ── Smoke-test runner (only runs when invoked directly) ──────────────────────
if __name__ == "__main__":
    print("SUTURE_TYPES:")
    for name, info in SUTURE_TYPES.items():
        n_bites = len(info["bites"](0.8, 1.0))
        print(f"  {name:36s}  {info['n_stitches']} stitches × {info['n_bites_per_stitch']} bites = {n_bites} targets")

    print("\nRendering gallery...")
    fig = render_gallery_grid()
    fig.savefig("/workspace/suture_gallery.png", dpi=80, bbox_inches="tight")
    print("  saved /workspace/suture_gallery.png")

    print("\nScore test:")
    score, dist, status = score_stitch((0.52, 0.30), (0.50, 0.30))
    print(f"  Near miss: score={score:.1f} dist={dist:.3f} status={status}")
    score, dist, status = score_stitch((0.50, 0.50), (0.50, 0.30))
    print(f"  Far:      score={score:.1f} dist={dist:.3f} status={status}")
