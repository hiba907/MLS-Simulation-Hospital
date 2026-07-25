"""Surgery Simulation Module — Wound tracking with markers + Suture practice.

Teaches students:
  1. How markers are placed on a wound (skin incision/cut) to track wound gap
  2. Instrument marker tracking relative to the wound
  3. Suture technique selection and click-based simulation
     (Simple Interrupted, Simple Continuous, Figure-8, Horizontal Mattress,
      Vertical Mattress, Subcuticular/Subcutaneous)
"""

import streamlit as st
import numpy as np
import math
import io
import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch, Arc
    MPL_OK = True
except ImportError:
    MPL_OK = False

from shared_session import dept_header, record_session

# ─────────────────────────────────────────────────────────────────────────────
# SUTURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

SUTURE_TYPES = {
    "Simple Interrupted": {
        "icon": "🪡",
        "description": (
            "Individual, separate stitches placed across the wound. "
            "Each stitch is tied independently. If one fails, the others hold."
        ),
        "indications": "Clean lacerations, skin closure, precise tension control.",
        "steps": [
            "Enter skin 5-8 mm from wound edge (right side)",
            "Pass needle through dermis in a CURVED arc beneath the wound",
            "Exit skin 5-8 mm from wound edge (left side), symmetrically",
            "Pull suture through, leaving a 3 cm tail",
            "Tie a surgeon's knot (2 throws forward, 1 back, 1 forward)",
            "Cut both ends leaving 3-5 mm tails",
            "Repeat every 5-8 mm along the wound",
        ],
        "color": "#2196F3",
        "stitches": "interrupted",
        "spacing_mm": 6,
    },
    "Simple Continuous": {
        "icon": "🔄",
        "description": (
            "A single running stitch along the entire wound. "
            "Faster than interrupted; if suture breaks, entire wound may open."
        ),
        "indications": "Long lacerations, subcutaneous layer closure, quick closure.",
        "steps": [
            "Place one anchoring simple interrupted stitch at one end",
            "Without cutting, continue placing stitches at 5 mm intervals",
            "Keep even tension throughout (don't pull too tight — causes ischemia)",
            "At the far end, create a loop and tie to the last loop (Aberdeen knot)",
        ],
        "color": "#4CAF50",
        "stitches": "continuous",
        "spacing_mm": 5,
    },
    "Figure-8": {
        "icon": "8️⃣",
        "description": (
            "A stitch that crosses itself forming a figure-8. Provides strong "
            "tension for high-stress areas. Useful for fascia and deep tissue."
        ),
        "indications": "Fascial closure, high-tension wounds, tendon repair.",
        "steps": [
            "Enter skin 8 mm from wound edge (right side)",
            "Exit on the left side, 8 mm from edge",
            "Re-enter same side 5 mm forward (crossing to right)",
            "Exit right side 5 mm forward",
            "Tie: this creates the figure-8 cross pattern",
        ],
        "color": "#FF9800",
        "stitches": "figure8",
        "spacing_mm": 10,
    },
    "Horizontal Mattress": {
        "icon": "↔️",
        "description": (
            "Two parallel bites across the wound, tied on the same side. "
            "Excellent for everting wound edges — keeps edges raised for better healing."
        ),
        "indications": "Skin under tension, areas prone to inversion (scalp, sole of foot).",
        "steps": [
            "Enter skin 8 mm from right wound edge",
            "Cross wound subcutaneously, exit left side",
            "Move 5 mm along wound SAME direction (still on left)",
            "Re-enter left skin, cross back subcutaneously",
            "Exit right side, 5 mm along from entry",
            "Tie on the right side — edges should evert slightly",
        ],
        "color": "#9C27B0",
        "stitches": "horizontal_mattress",
        "spacing_mm": 12,
    },
    "Vertical Mattress": {
        "icon": "↕️",
        "description": (
            "Two bites: one deep (far from edge) and one shallow (close to edge). "
            "Eliminates dead space AND everts skin — best of both worlds."
        ),
        "indications": "Deep lacerations, areas prone to dead space (axilla, groin).",
        "steps": [
            "Enter skin FAR from right wound edge (10-12 mm)",
            "Deep pass beneath wound, exit FAR from left wound edge",
            "Re-enter LEFT side CLOSE to wound edge (3-4 mm)",
            "Shallow pass, exit RIGHT side CLOSE to wound edge",
            "Tie on the right — note the far-near-near-far pattern",
        ],
        "color": "#F44336",
        "stitches": "vertical_mattress",
        "spacing_mm": 12,
    },
    "Subcuticular (Running)": {
        "icon": "🎽",
        "description": (
            "Runs horizontally WITHIN the dermis — no visible entry/exit holes "
            "on skin surface. Leaves minimal scarring. Cannot be removed easily if infected."
        ),
        "indications": "Cosmetic closure (face, neck), clean surgical incisions.",
        "steps": [
            "Anchor stitch at one end BELOW skin surface",
            "Enter dermis at wound edge — run needle horizontally",
            "Alternate sides: left dermis then right dermis, at same depth",
            "Keep bites 5 mm long, advance 5 mm per pass",
            "At far end, exit through skin or anchor with external knot",
            "Tape reinforcement strips (Steri-strips) on top",
        ],
        "color": "#00BCD4",
        "stitches": "subcuticular",
        "spacing_mm": 5,
    },
}

# Window/level for wound visualization
WOUND_COLORS = {
    "skin": "#D2956A",
    "dermis": "#C4735A",
    "subcutaneous": "#F4D1A0",
    "muscle": "#C84B3C",
    "marker_wound": "#FF3333",
    "marker_instrument": "#3388FF",
    "suture": "#F5DEB3",
    "suture_dark": "#8B6914",
}


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_wound_cross_section(ax, wound_length_mm=30, wound_depth_mm=8,
                               wound_width_mm=5, show_markers=True,
                               num_markers_per_side=3, suture_type=None,
                               placed_sutures=None):
    """Draw a realistic skin wound cross-section with markers and optional sutures."""
    ax.set_aspect('equal')
    ax.set_xlim(-wound_length_mm / 2 - 15, wound_length_mm / 2 + 15)
    ax.set_ylim(-wound_depth_mm - 8, 18)
    ax.set_facecolor('#1a1a1a')

    # Skin layers (drawn as horizontal bands)
    # Epidermis + dermis (surface)
    skin_rect = patches.Rectangle(
        (-wound_length_mm / 2 - 12, 0), wound_length_mm + 24, 6,
        linewidth=0, facecolor=WOUND_COLORS["skin"], alpha=0.9, zorder=2
    )
    ax.add_patch(skin_rect)

    # Dermis
    dermis_rect = patches.Rectangle(
        (-wound_length_mm / 2 - 12, -3), wound_length_mm + 24, 4,
        linewidth=0, facecolor=WOUND_COLORS["dermis"], alpha=0.85, zorder=2
    )
    ax.add_patch(dermis_rect)

    # Subcutaneous fat
    fat_rect = patches.Rectangle(
        (-wound_length_mm / 2 - 12, -wound_depth_mm - 2), wound_length_mm + 24, wound_depth_mm,
        linewidth=0, facecolor=WOUND_COLORS["subcutaneous"], alpha=0.7, zorder=1
    )
    ax.add_patch(fat_rect)

    # Wound gap (the cut — remove skin in the gap)
    gap = wound_width_mm / 2
    wound_left_x = -gap
    wound_right_x = gap

    # White out the wound gap
    wound_gap = patches.Rectangle(
        (wound_left_x, -wound_depth_mm), wound_width_mm, wound_depth_mm + 6,
        linewidth=0, facecolor='#1a1a1a', zorder=3
    )
    ax.add_patch(wound_gap)

    # Draw wound edges (beveled cut lines)
    ax.plot([wound_left_x, wound_left_x - 1], [6, -wound_depth_mm],
            color='#8B3A3A', linewidth=2.5, zorder=4)
    ax.plot([wound_right_x, wound_right_x + 1], [6, -wound_depth_mm],
            color='#8B3A3A', linewidth=2.5, zorder=4)

    # Blood / wound floor
    ax.fill_between(
        [wound_left_x, wound_right_x],
        [-wound_depth_mm], [-wound_depth_mm + 1],
        color='#8B0000', alpha=0.6, zorder=4
    )

    # ── MARKERS ON WOUND EDGES ──
    if show_markers:
        half = wound_length_mm / 2
        # Determine marker positions along wound edges
        if num_markers_per_side == 2:
            # Small wound: 2 markers each side, just outside the wound
            positions = [-half * 0.4, half * 0.4]
        elif num_markers_per_side == 3:
            positions = [-half * 0.6, 0, half * 0.6]
        else:
            positions = list(np.linspace(-half * 0.8, half * 0.8, num_markers_per_side))

        marker_size = 3.5
        for i, pos in enumerate(positions):
            # Left side marker (red square ArUco-style)
            left_marker = patches.FancyBboxPatch(
                (wound_left_x - marker_size - 2, pos - marker_size / 2 - 2),
                marker_size, marker_size,
                boxstyle="square,pad=0.2",
                linewidth=1.5, edgecolor='white',
                facecolor=WOUND_COLORS["marker_wound"], zorder=10, alpha=0.95
            )
            ax.add_patch(left_marker)
            # Inner white/black pattern (simplified ArUco)
            ax.text(wound_left_x - marker_size / 2 - 2, pos - 1.5, f"L{i+1}",
                    ha='center', va='center', fontsize=4, color='white',
                    fontweight='bold', zorder=11)

            # Right side marker
            right_marker = patches.FancyBboxPatch(
                (wound_right_x + 2, pos - marker_size / 2 - 2),
                marker_size, marker_size,
                boxstyle="square,pad=0.2",
                linewidth=1.5, edgecolor='white',
                facecolor=WOUND_COLORS["marker_wound"], zorder=10, alpha=0.95
            )
            ax.add_patch(right_marker)
            ax.text(wound_right_x + marker_size / 2 + 2, pos - 1.5, f"R{i+1}",
                    ha='center', va='center', fontsize=4, color='white',
                    fontweight='bold', zorder=11)

            # Dashed line between corresponding markers (shows gap measurement)
            ax.annotate('', xy=(wound_right_x + 2, pos - 1),
                        xytext=(wound_left_x - 2, pos - 1),
                        arrowprops=dict(arrowstyle='<->', color='#FFD700',
                                        lw=1.5, linestyle='dashed'), zorder=9)

        # Gap label
        ax.text(0, positions[len(positions) // 2] + 1.5, f"{wound_width_mm:.1f} mm gap",
                ha='center', va='bottom', fontsize=7, color='#FFD700', fontweight='bold', zorder=12)

        # Instrument marker (blue, at top of wound)
        instr_x = 0
        instr_marker = patches.FancyBboxPatch(
            (instr_x - 2, 9), 4, 4,
            boxstyle="square,pad=0.2",
            linewidth=1.5, edgecolor='white',
            facecolor=WOUND_COLORS["marker_instrument"], zorder=10
        )
        ax.add_patch(instr_marker)
        ax.text(instr_x, 11, "INS", ha='center', va='center',
                fontsize=5, color='white', fontweight='bold', zorder=11)

        # Arrow from instrument marker to wound
        ax.annotate('', xy=(0, 6), xytext=(0, 9),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5), zorder=9)

    # ── SUTURES (if any placed) ──
    if placed_sutures and suture_type:
        _draw_sutures_cross_section(ax, suture_type, placed_sutures, wound_left_x,
                                     wound_right_x, wound_depth_mm)

    # Labels
    ax.text(-wound_length_mm / 2 - 12, 4, "Epidermis", fontsize=6.5, color='#111',
            va='center', ha='left', fontweight='bold')
    ax.text(-wound_length_mm / 2 - 12, -1.5, "Dermis", fontsize=6.5, color='#fff',
            va='center', ha='left', fontweight='bold')
    ax.text(-wound_length_mm / 2 - 12, -wound_depth_mm / 2 - 1, "Subcut.",
            fontsize=6.5, color='#555', va='center', ha='left')

    ax.set_title("Wound Cross-Section View", color='white', fontsize=10, pad=6)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    ax.set_xlabel("Position (mm)", color='#aaa', fontsize=8)
    ax.set_ylabel("Depth (mm)", color='#aaa', fontsize=8)


def _draw_sutures_cross_section(ax, suture_type, placed_sutures, wound_left, wound_right, depth):
    """Draw suture patterns in cross-section on the wound."""
    stype = SUTURE_TYPES[suture_type]["stitches"]
    clr = SUTURE_TYPES[suture_type]["color"]
    dark = WOUND_COLORS["suture_dark"]

    for sx in placed_sutures:
        if stype == "interrupted":
            # Enter right, arc under, exit left
            xs = [wound_right + 5, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 5]
            ys = [5, 0, -depth * 0.7, 0, 5]
            ax.plot(xs, ys, color=clr, linewidth=2, zorder=8, solid_capstyle='round')
            # Knot dot on skin
            ax.plot(0, 6, 'o', color=dark, markersize=4, zorder=9)

        elif stype == "continuous":
            xs = [wound_right + 4, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 4]
            ys = [4, 0, -depth * 0.6, 0, 4]
            ax.plot(xs, ys, color=clr, linewidth=1.8, zorder=8, alpha=0.85)

        elif stype == "figure8":
            # First bite
            ax.plot([wound_right + 6, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 6],
                    [5, 0, -depth * 0.8, 0, 5], color=clr, linewidth=2, zorder=8)
            # Second bite (crossing)
            ax.plot([wound_right + 6, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 6],
                    [5, -2, -depth * 0.5, -2, 5], color=clr, linewidth=2, zorder=8, alpha=0.7)
            ax.plot(0, 6, 'X', color=dark, markersize=5, zorder=9)

        elif stype == "horizontal_mattress":
            # Bite 1 (right to left)
            ax.plot([wound_right + 7, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 7],
                    [5, 0, -depth * 0.7, 0, 5], color=clr, linewidth=2, zorder=8)
            # Return bite (offset)
            ax.plot([wound_left - 7, wound_left - 0.5, 0, wound_right + 0.5, wound_right + 7],
                    [3, 0, -depth * 0.5, 0, 3], color=clr, linewidth=2, zorder=8, alpha=0.7)
            ax.plot(wound_right + 7, 4, 'o', color=dark, markersize=4, zorder=9)

        elif stype == "vertical_mattress":
            # Deep far bite
            ax.plot([wound_right + 10, wound_right + 0.5, 0, wound_left - 0.5, wound_left - 10],
                    [5, 0, -depth * 0.9, 0, 5], color=clr, linewidth=2, zorder=8)
            # Shallow near bite (reverse direction)
            ax.plot([wound_left - 3, wound_left - 0.3, 0, wound_right + 0.3, wound_right + 3],
                    [5, -1, -depth * 0.2, -1, 5], color=clr, linewidth=2, zorder=8, alpha=0.8)
            ax.plot(wound_right + 10, 5, 'o', color=dark, markersize=4, zorder=9)

        elif stype == "subcuticular":
            # Horizontal passes within dermis (no surface exit)
            ax.plot([wound_left - 0.5, 0, wound_right + 0.5],
                    [-1.5, -2.0, -1.5], color=clr, linewidth=2, zorder=8)
            ax.plot([wound_right + 0.5, 0, wound_left - 0.5],
                    [-2.5, -3.0, -2.5], color=clr, linewidth=2, zorder=8, alpha=0.8)


def _draw_wound_top_view(ax, wound_length_mm=30, wound_width_mm=5,
                          show_markers=True, num_markers_per_side=3,
                          placed_sutures=None, suture_type=None):
    """Top-down view of wound with markers and sutures."""
    ax.set_aspect('equal')
    ax.set_xlim(-20, wound_length_mm + 20)
    ax.set_ylim(-12, 12)
    ax.set_facecolor('#C8956A')  # skin color

    # Wound gap
    wound_rect = patches.Rectangle(
        (0, -wound_width_mm / 2), wound_length_mm, wound_width_mm,
        linewidth=0, facecolor='#1a0000', zorder=3
    )
    ax.add_patch(wound_rect)

    # Wound edges (dark red lines)
    ax.plot([0, wound_length_mm], [-wound_width_mm / 2, -wound_width_mm / 2],
            color='#8B0000', linewidth=2.5, zorder=4)
    ax.plot([0, wound_length_mm], [wound_width_mm / 2, wound_width_mm / 2],
            color='#8B0000', linewidth=2.5, zorder=4)

    # ── MARKERS on wound edges ──
    if show_markers:
        if num_markers_per_side == 2:
            positions = [wound_length_mm * 0.25, wound_length_mm * 0.75]
        elif num_markers_per_side == 3:
            positions = [wound_length_mm * 0.2, wound_length_mm * 0.5, wound_length_mm * 0.8]
        else:
            positions = list(np.linspace(wound_length_mm * 0.1, wound_length_mm * 0.9,
                                          num_markers_per_side))

        marker_sz = 2.5
        for i, px in enumerate(positions):
            # Lower edge markers
            lo_marker = patches.FancyBboxPatch(
                (px - marker_sz / 2, -wound_width_mm / 2 - marker_sz - 1.5),
                marker_sz, marker_sz,
                boxstyle="square,pad=0.1",
                linewidth=1, edgecolor='white',
                facecolor=WOUND_COLORS["marker_wound"], zorder=10
            )
            ax.add_patch(lo_marker)
            ax.text(px, -wound_width_mm / 2 - marker_sz / 2 - 1.5, f"L{i+1}",
                    ha='center', va='center', fontsize=5, color='white', fontweight='bold')

            # Upper edge markers
            up_marker = patches.FancyBboxPatch(
                (px - marker_sz / 2, wound_width_mm / 2 + 1.5),
                marker_sz, marker_sz,
                boxstyle="square,pad=0.1",
                linewidth=1, edgecolor='white',
                facecolor=WOUND_COLORS["marker_wound"], zorder=10
            )
            ax.add_patch(up_marker)
            ax.text(px, wound_width_mm / 2 + marker_sz / 2 + 1.5, f"R{i+1}",
                    ha='center', va='center', fontsize=5, color='white', fontweight='bold')

            # Gap line
            ax.plot([px, px], [-wound_width_mm / 2, wound_width_mm / 2],
                    color='#FFD700', linestyle=':', linewidth=1, alpha=0.7, zorder=5)

        # Instrument marker
        ax.plot(wound_length_mm * 0.5, 9, 's', color=WOUND_COLORS["marker_instrument"],
                markersize=10, zorder=11)
        ax.text(wound_length_mm * 0.5, 9, "INS",
                ha='center', va='center', fontsize=5, color='white', fontweight='bold')
        ax.annotate('', xy=(wound_length_mm * 0.5, wound_width_mm / 2 + 1),
                    xytext=(wound_length_mm * 0.5, 7.5),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # ── SUTURES (top view) ──
    if placed_sutures and suture_type:
        stype = SUTURE_TYPES[suture_type]["stitches"]
        clr = SUTURE_TYPES[suture_type]["color"]
        for sx in placed_sutures:
            if stype == "interrupted":
                ax.plot([sx, sx], [-wound_width_mm / 2 - 4, wound_width_mm / 2 + 4],
                        color=clr, linewidth=2, zorder=8)
                ax.plot(sx, 0, 'o', color='#8B6914', markersize=3, zorder=9)
            elif stype == "continuous":
                ax.plot([sx, sx], [-wound_width_mm / 2 - 3, wound_width_mm / 2 + 3],
                        color=clr, linewidth=1.8, zorder=8, alpha=0.85)
            elif stype == "figure8":
                ax.plot([sx - 2, sx + 2], [-wound_width_mm / 2 - 4, wound_width_mm / 2 + 4],
                        color=clr, linewidth=2, zorder=8)
                ax.plot([sx + 2, sx - 2], [-wound_width_mm / 2 - 4, wound_width_mm / 2 + 4],
                        color=clr, linewidth=2, zorder=8, alpha=0.6)
            elif stype in ("horizontal_mattress", "vertical_mattress"):
                ax.plot([sx, sx], [-wound_width_mm / 2 - 5, wound_width_mm / 2 + 5],
                        color=clr, linewidth=2.5, zorder=8)
                ax.plot([sx - 2, sx + 2], [-wound_width_mm / 2 - 5, -wound_width_mm / 2 - 5],
                        color=clr, linewidth=2, zorder=8)
                ax.plot([sx - 2, sx + 2], [wound_width_mm / 2 + 5, wound_width_mm / 2 + 5],
                        color=clr, linewidth=2, zorder=8)
            elif stype == "subcuticular":
                # Diagonal passes (not visible on surface in real life, show dashed)
                ax.plot([sx - 1, sx + 1], [-wound_width_mm / 2, wound_width_mm / 2],
                        color=clr, linewidth=1.5, linestyle='--', zorder=8, alpha=0.7)

    # Draw connecting thread for continuous sutures
    if placed_sutures and suture_type and len(placed_sutures) > 1:
        stype = SUTURE_TYPES[suture_type]["stitches"]
        if stype in ("continuous", "subcuticular"):
            sorted_sx = sorted(placed_sutures)
            ax.plot(sorted_sx, [-wound_width_mm / 2 - 3] * len(sorted_sx),
                    color=SUTURE_TYPES[suture_type]["color"], linewidth=1.5,
                    linestyle='-', zorder=7, alpha=0.6)

    ax.set_title("Top-Down Wound View", color='#222', fontsize=10, pad=6)
    ax.tick_params(colors='#444')
    ax.set_xlabel("Along wound (mm)", color='#444', fontsize=8)
    ax.set_ylabel("Width (mm)", color='#444', fontsize=8)


def _make_suture_diagram(suture_name):
    """Generate a detailed educational diagram for a suture technique."""
    info = SUTURE_TYPES[suture_name]
    stype = info["stitches"]
    clr = info["color"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('#0f1117')

    # --- Left panel: technique close-up ---
    ax = axes[0]
    ax.set_facecolor('#1e1e2e')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')

    # Skin block
    skin = patches.Rectangle((-18, 0), 36, 8, facecolor='#D2956A', linewidth=0, zorder=1)
    derm = patches.Rectangle((-18, -3), 36, 3, facecolor='#C4735A', linewidth=0, zorder=1)
    subc = patches.Rectangle((-18, -12), 36, 9, facecolor='#F4D1A0', linewidth=0, zorder=1)
    ax.add_patch(skin)
    ax.add_patch(derm)
    ax.add_patch(subc)

    # Wound gap
    gap = patches.Rectangle((-2.5, -12), 5, 11, facecolor='#1e1e2e', linewidth=0, zorder=2)
    ax.add_patch(gap)
    ax.plot([-2.5, -2.5], [8, -12], color='#8B0000', linewidth=2, zorder=3)
    ax.plot([2.5, 2.5], [8, -12], color='#8B0000', linewidth=2, zorder=3)

    # Draw suture path
    if stype == "interrupted":
        xs = [10, 2.5, 0, -2.5, -10]
        ys = [7, -1, -9, -1, 7]
        ax.plot(xs, ys, color=clr, linewidth=2.5, zorder=8, solid_capstyle='round')
        ax.plot(0, 8.5, 'o', color='#8B6914', markersize=6, zorder=9)
        ax.plot(0, 8.5, 'o', color='#F5DEB3', markersize=3, zorder=10)
        # Entry/exit annotations
        ax.annotate('1', xy=(10, 7), fontsize=9, color='yellow', ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='black', alpha=0.7), zorder=11)
        ax.annotate('2', xy=(-10, 7), fontsize=9, color='yellow', ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='black', alpha=0.7), zorder=11)

    elif stype == "continuous":
        # Multiple stitches
        for i, ox in enumerate([-8, 0, 8]):
            xs = [ox + 5, 2.5, 0, -2.5, ox - 5]
            ys = [7, -1, -7, -1, 7]
            ax.plot(xs, ys, color=clr, linewidth=2, zorder=8, alpha=0.85)
        # Running thread along top
        ax.plot([-13, -8, -3, 3, 8, 13], [7.5, 7.5, 7.5, 7.5, 7.5, 7.5],
                color=clr, linewidth=1.5, zorder=7, linestyle='--', alpha=0.6)

    elif stype == "figure8":
        # Two bites crossing
        ax.plot([12, 2.5, 0, -2.5, -12], [7, 0, -10, 0, 7],
                color=clr, linewidth=2.5, zorder=8)
        ax.plot([12, 2.5, 0, -2.5, -12], [7, -2, -6, -2, 7],
                color=clr, linewidth=2.5, zorder=8, alpha=0.6)
        ax.plot(0, 8.5, 'X', color='#8B6914', markersize=8, zorder=9)
        ax.annotate('8', xy=(0, -4), fontsize=14, color='yellow', ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='black', alpha=0.5), zorder=11)

    elif stype == "horizontal_mattress":
        # Bite 1 (right to left)
        ax.plot([13, 2.5, 0, -2.5, -13], [7, 0, -9, 0, 7], color=clr, linewidth=2.5, zorder=8)
        # Return on surface
        ax.plot([-13, -13], [7, 4], color=clr, linewidth=2, zorder=8)
        ax.plot([-13, 13], [4, 4], color=clr, linewidth=1.8, zorder=7, linestyle='dashed',
                alpha=0.5)
        ax.plot([13, 13], [4, 7], color=clr, linewidth=2, zorder=8)
        ax.plot(13, 7, 'o', color='#8B6914', markersize=5, zorder=9)
        ax.annotate('<->', xy=(0, 5.5), fontsize=10, color='yellow', ha='center', zorder=11)

    elif stype == "vertical_mattress":
        # Deep far bite
        ax.plot([14, 2.5, 0, -2.5, -14], [7, 0, -11, 0, 7], color=clr, linewidth=2.5, zorder=8)
        # Shallow near return
        ax.plot([-4, -2.5, 0, 2.5, 4], [7, -1, -3, -1, 7],
                color=clr, linewidth=2, zorder=8, alpha=0.8)
        ax.plot(14, 7, 'o', color='#8B6914', markersize=5, zorder=9)
        ax.annotate('FAR', xy=(14, 9), fontsize=7, color='yellow', ha='center')
        ax.annotate('near', xy=(4, 9), fontsize=7, color='yellow', ha='center')

    elif stype == "subcuticular":
        # Horizontal passes in dermis
        for i, dy in enumerate([-1, -2, -3]):
            direction = 1 if i % 2 == 0 else -1
            ax.plot([direction * -12, direction * 12], [dy, dy - 0.5],
                    color=clr, linewidth=2, zorder=8, alpha=0.9)
            ax.plot([direction * 12, direction * 12], [dy, dy - 0.5],
                    color=clr, linewidth=1.5, zorder=8)
        ax.annotate('WITHIN DERMIS', xy=(0, -1.5), fontsize=7, color='yellow',
                    ha='center', zorder=11)

    ax.set_title(f"{info['icon']} {suture_name}\n(Technique View)", color='white', fontsize=10)
    ax.axis('off')

    # --- Right panel: steps + info ---
    ax2 = axes[1]
    ax2.set_facecolor('#0f1117')
    ax2.axis('off')

    y = 0.97
    ax2.text(0.05, y, f"{info['icon']} {suture_name}", transform=ax2.transAxes,
             fontsize=13, fontweight='bold', color=clr, va='top')
    y -= 0.10
    ax2.text(0.05, y, info['description'], transform=ax2.transAxes,
             fontsize=8.5, color='#ccc', va='top', wrap=True,
             multialignment='left')
    y -= 0.12
    ax2.text(0.05, y, "Indications:", transform=ax2.transAxes,
             fontsize=9, fontweight='bold', color='#4CAF50', va='top')
    y -= 0.07
    ax2.text(0.05, y, info['indications'], transform=ax2.transAxes,
             fontsize=8, color='#aaa', va='top')
    y -= 0.10
    ax2.text(0.05, y, "Steps:", transform=ax2.transAxes,
             fontsize=9, fontweight='bold', color='#FF9800', va='top')
    y -= 0.07
    for step in info['steps']:
        ax2.text(0.05, y, f"  - {step}", transform=ax2.transAxes,
                 fontsize=7.5, color='#ddd', va='top')
        y -= 0.075
        if y < 0.02:
            break

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MARKER EXPLANATION DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def _draw_marker_explanation():
    """Educational diagram explaining how markers work on a wound."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')

    titles = [
        "Small Wound (< 2 cm)\n2 markers per edge",
        "Medium Wound (2-5 cm)\n3 markers per edge",
        "Large Wound (> 5 cm)\n4+ markers per edge",
    ]
    num_markers = [2, 3, 4]
    wound_lengths = [15, 30, 50]

    for ax, title, nm, wl in zip(axes, titles, num_markers, wound_lengths):
        _draw_wound_top_view(ax, wound_length_mm=wl, wound_width_mm=5,
                              show_markers=True, num_markers_per_side=nm)
        ax.set_title(title, color='white', fontsize=9, pad=4)
        ax.set_facecolor('#2a2a2a')

    fig.suptitle("Wound Marker Placement Strategy", color='white', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODULE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run():
    dept_header("Surgery Simulation", "🔪",
                "Wound tracking with optical markers & suture technique practice")

    if not MPL_OK:
        st.error("matplotlib is required for this module. Run: pip install matplotlib")
        return

    tabs = st.tabs(["🩹 Wound & Markers", "🪡 Suture Practice", "📊 My Progress"])

    # ─── TAB 1: WOUND & MARKERS ───────────────────────────────────────────────
    with tabs[0]:
        st.markdown("## Wound Anatomy & Marker Tracking")
        st.markdown(
            """
A **wound** (surgical incision or traumatic laceration) is a **break in the skin** — it can be:
- **Small** (< 2 cm): single stitch or adhesive strips may suffice
- **Medium** (2-5 cm): standard suture repair
- **Large** (> 5 cm): layered closure, possible drain

### How are markers placed on the wound?

Optical markers (ArUco / fiducial) are attached **along both edges** of the wound:
- **Red markers (L1, L2, ...)** on the left/lower wound edge
- **Red markers (R1, R2, ...)** on the right/upper wound edge
- **Blue marker (INS)** on the surgical instrument

The system measures the **gap** between corresponding L/R pairs to track:
- How open the wound is before suturing
- Whether sutures are correctly approximating the edges
- Instrument position relative to the wound center
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            wound_length = st.slider("Wound length (mm)", 10, 80, 30, 5)
        with col2:
            wound_width = st.slider("Wound gap/width (mm)", 2, 20, 5, 1)
        with col3:
            wound_size_cat = "Small" if wound_length < 20 else ("Medium" if wound_length < 50 else "Large")
            n_markers = 2 if wound_length < 20 else (3 if wound_length < 50 else 4)
            st.metric("Category", wound_size_cat)
            st.metric("Markers per side", n_markers)

        st.markdown("### Marker Placement Diagram")

        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
        fig1.patch.set_facecolor('#0f1117')

        _draw_wound_cross_section(
            axes1[0], wound_length_mm=wound_length, wound_depth_mm=8,
            wound_width_mm=wound_width, show_markers=True,
            num_markers_per_side=n_markers
        )
        _draw_wound_top_view(
            axes1[1], wound_length_mm=wound_length, wound_width_mm=wound_width,
            show_markers=True, num_markers_per_side=n_markers
        )

        fig1.tight_layout()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=130, bbox_inches='tight',
                     facecolor='#0f1117')
        buf1.seek(0)
        st.image(buf1, use_column_width=True)
        plt.close(fig1)

        st.markdown("### All Wound Sizes — Marker Strategy")
        fig_explain = _draw_marker_explanation()
        buf_ex = io.BytesIO()
        fig_explain.savefig(buf_ex, format='png', dpi=120, bbox_inches='tight',
                             facecolor='#0f1117')
        buf_ex.seek(0)
        st.image(buf_ex, use_column_width=True)
        plt.close(fig_explain)

        with st.expander("Technical Details — How Marker Tracking Works"):
            st.markdown("""
**For SMALL wounds (< 2 cm):**
- 2 markers per edge (total 4 wound markers + 1 instrument marker)
- Markers placed at 25% and 75% of wound length

**For MEDIUM wounds (2-5 cm):**
- 3 markers per edge (total 6 wound markers + 1 instrument marker)
- Placed at 20%, 50%, 80% of wound length
- Middle pair gives the critical gap measurement

**For LARGE wounds (> 5 cm):**
- 4+ markers per edge
- Evenly spaced along wound length
- Multiple gap measurements identify areas of incomplete closure

**Tracking algorithm:**
1. Camera sees all markers in real time
2. Corresponding L/R pairs are matched by index (L1-R1, L2-R2, ...)
3. Gap = Euclidean distance between Ln and Rn in camera space
4. Instrument position = distance from INS marker to wound centerline
5. Score = improvement in gap closure after each suture
""")

    # ─── TAB 2: SUTURE PRACTICE ───────────────────────────────────────────────
    with tabs[1]:
        st.markdown("## Suture Technique Practice")
        st.markdown("Select a suture type to learn the technique, then practice placing sutures on the simulated wound.")

        col_sel, col_wound = st.columns([1, 2])

        with col_sel:
            st.markdown("### Choose Suture Type")
            selected_suture = st.radio(
                "Suture technique:",
                list(SUTURE_TYPES.keys()),
                format_func=lambda x: f"{SUTURE_TYPES[x]['icon']} {x}",
                key="suture_type_radio"
            )
            info = SUTURE_TYPES[selected_suture]
            st.caption(info['description'])
            st.caption(f"**Use for:** {info['indications']}")

        with col_wound:
            st.markdown("### Wound Settings")
            w_len = st.slider("Wound length (mm)", 15, 60, 30, 5, key="prac_len")
            w_width = st.slider("Starting wound gap (mm)", 2, 15, 6, 1, key="prac_width")

        # Initialize suture placement session state
        if "placed_sutures" not in st.session_state:
            st.session_state.placed_sutures = []
        if "current_wound_len" not in st.session_state:
            st.session_state.current_wound_len = w_len

        # Reset if wound length changed
        if st.session_state.current_wound_len != w_len:
            st.session_state.placed_sutures = []
            st.session_state.current_wound_len = w_len

        # Suture technique diagram
        st.markdown("### Technique Diagram")
        fig_diag = _make_suture_diagram(selected_suture)
        buf_diag = io.BytesIO()
        fig_diag.savefig(buf_diag, format='png', dpi=130, bbox_inches='tight',
                          facecolor='#0f1117')
        buf_diag.seek(0)
        st.image(buf_diag, use_column_width=True)
        plt.close(fig_diag)

        st.markdown("---")
        st.markdown("### Practice: Place Sutures on the Wound")

        # Suture placement slider
        spacing = SUTURE_TYPES[selected_suture]["spacing_mm"]
        new_suture_pos = st.slider(
            f"Position new suture along wound (mm from start)",
            0.0, float(w_len), float(w_len / 2), float(spacing / 2),
            key="suture_pos_slider"
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Place Suture Here", key="place_suture_btn"):
                if new_suture_pos not in st.session_state.placed_sutures:
                    st.session_state.placed_sutures.append(new_suture_pos)
                    st.success(f"Suture placed at {new_suture_pos:.0f} mm")
        with col_btn2:
            if st.button("Remove Last Suture", key="remove_suture_btn"):
                if st.session_state.placed_sutures:
                    removed = st.session_state.placed_sutures.pop()
                    st.info(f"Removed suture at {removed:.0f} mm")
        with col_btn3:
            if st.button("Clear All Sutures", key="clear_sutures_btn"):
                st.session_state.placed_sutures = []
                st.info("All sutures cleared")

        # Score
        placed = st.session_state.placed_sutures
        n_placed = len(placed)
        ideal_count = max(1, int(w_len / spacing))

        # Calculate wound closure score
        if n_placed > 0:
            # Each suture reduces gap
            reduction_per_suture = 1.0 / (ideal_count + 1)
            gap_remaining = max(0.5, w_width * (1 - min(1.0, n_placed * reduction_per_suture)))
            score = min(100, int(100 * n_placed / ideal_count))

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Sutures placed", n_placed, f"target: ~{ideal_count}")
            col_m2.metric("Wound gap (mm)", f"{gap_remaining:.1f}", f"-{w_width - gap_remaining:.1f}")
            col_m3.metric("Closure score", f"{score}%",
                          "Good" if score >= 80 else ("More needed" if score >= 50 else "Continue"))
        else:
            gap_remaining = float(w_width)
            score = 0

        # Draw the wound with placed sutures
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        fig2.patch.set_facecolor('#0f1117')

        _draw_wound_cross_section(
            axes2[0], wound_length_mm=w_len, wound_depth_mm=8,
            wound_width_mm=gap_remaining, show_markers=True,
            num_markers_per_side=2 if w_len < 20 else 3,
            placed_sutures=placed, suture_type=selected_suture
        )
        _draw_wound_top_view(
            axes2[1], wound_length_mm=w_len, wound_width_mm=gap_remaining,
            show_markers=True, num_markers_per_side=2 if w_len < 20 else 3,
            placed_sutures=placed, suture_type=selected_suture
        )

        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=130, bbox_inches='tight',
                     facecolor='#0f1117')
        buf2.seek(0)
        st.image(buf2, use_column_width=True)
        plt.close(fig2)

        # Step-by-step guidance
        if n_placed > 0 and score < 100:
            step_idx = min(n_placed, len(info["steps"]) - 1)
            st.info(f"Next step: {info['steps'][step_idx]}")
        elif score >= 80:
            st.success(f"Excellent closure! Score: {score}%. The wound is well approximated.")
            # Record session
            if st.button("Save to My Progress", key="save_surgery"):
                record_session(
                    dept="Surgery",
                    score=score,
                    details={
                        "suture_type": selected_suture,
                        "wound_length_mm": w_len,
                        "sutures_placed": n_placed,
                        "gap_remaining_mm": round(gap_remaining, 1),
                    }
                )
                st.success("Progress saved!")

    # ─── TAB 3: PROGRESS ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("## My Surgery Progress")
        surgery_sessions = st.session_state.get("sessions", {}).get("Surgery", [])
        if not surgery_sessions:
            st.info("No surgery sessions recorded yet. Practice sutures in the Suture Practice tab!")
        else:
            import pandas as pd
            rows = []
            for s in surgery_sessions:
                rows.append({
                    "Date": s.get("ts", "")[:10],
                    "Score": s.get("score", 0),
                    "Suture": s.get("details", {}).get("suture_type", "-"),
                    "Wound (mm)": s.get("details", {}).get("wound_length_mm", "-"),
                    "Sutures": s.get("details", {}).get("sutures_placed", "-"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            if MPL_OK and len(rows) > 1:
                fig3, ax3 = plt.subplots(figsize=(8, 3))
                fig3.patch.set_facecolor('#0f1117')
                ax3.set_facecolor('#1e1e2e')
                ax3.plot([r["Score"] for r in rows], marker='o', color='#4CAF50', linewidth=2)
                ax3.set_title("Score over time", color='white')
                ax3.tick_params(colors='white')
                ax3.set_ylabel("Score (%)", color='white')
                ax3.set_ylim(0, 105)
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format='png', dpi=110, bbox_inches='tight',
                             facecolor='#0f1117')
                buf3.seek(0)
                st.image(buf3, use_column_width=True)
                plt.close(fig3)
