#!/usr/bin/env python3
"""Renders diagram/graph_diagram.png for the submission.

Run: python diagram/render_diagram.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

COLORS = {
    "entry": "#374151",
    "deterministic": "#2563eb",
    "model": "#7c3aed",
    "decision": "#b45309",
    "terminal": "#15803d",
    "fail": "#b91c1c",
}


def box(x, y, w, h, label, color, text_color="white", fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
             fontsize=fontsize, color=text_color, weight="bold", wrap=True)
    return (x, y, w, h)


def diamond(x, y, w, h, label, color, fontsize=9):
    pts = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
    poly = plt.Polygon(pts, closed=True, edgecolor=color, facecolor=color, alpha=0.92, linewidth=1.4)
    ax.add_patch(poly)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
             fontsize=fontsize, color="white", weight="bold")
    return (x, y, w, h)


def arrow(p1, p2, label=None, color="#374151", style="-", curve=0.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
        color=color, linestyle=style,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + curve * 1.2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=8.5, color=color, ha="center",
                 va="center", style="italic",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# Nodes -----------------------------------------------------------------
start = box(5.7, 7.2, 1.6, 0.5, "START\n(question)", COLORS["entry"])
triage = box(5.4, 6.1, 2.2, 0.7, "TRIAGE\n(deterministic rules)", COLORS["deterministic"])

tri_dec = diamond(5.55, 5.0, 1.9, 0.85, "classification?", COLORS["decision"], fontsize=8)

retrieval = box(5.4, 3.75, 2.2, 0.7, "RETRIEVAL\n(local embedding model)", COLORS["model"])
generation = box(5.4, 2.6, 2.2, 0.7, "GENERATION\n(local HF LLM)", COLORS["model"])
verification = box(5.4, 1.45, 2.2, 0.7, "VERIFICATION\n(schema + grounding + safety)", COLORS["deterministic"])

ver_dec = diamond(2.7, 1.3, 1.9, 0.85, "passed?", COLORS["decision"], fontsize=8)

retry = box(0.3, 2.6, 2.0, 0.7, "INCREMENT_RETRY\n(retry_count += 1, max 1)", COLORS["deterministic"], fontsize=8.5)

format_resp = box(9.4, 3.1, 2.4, 0.7, "FORMAT_RESPONSE\n(build schema-valid JSON)", COLORS["terminal"])
safe_fail = box(9.4, 0.6, 2.4, 0.7, "SAFE_FAILURE\n(retries exhausted)", COLORS["fail"])

end = box(9.9, 5.5, 1.4, 0.5, "END", COLORS["entry"])

# Edges -------------------------------------------------------------------
arrow((6.5, 7.2), (6.5, 6.8))
arrow((6.5, 6.1), (6.5, 5.85))

arrow((6.15, 5.0), (6.15, 4.45), label="answerable /\nrequires_escalation")
arrow((6.5, 5.4), (10.6, 5.4), label="out_of_scope /\nrequires_clarification", curve=0.15)
ax.add_patch(FancyArrowPatch((7.45, 5.4), (10.6, 5.4), arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.6, color="#374151", connectionstyle="arc3,rad=0.15"))
arrow((10.6, 5.5), (10.6, 6.0), curve=0.0)
arrow((10.6, 3.8), (10.6, 5.0))

arrow((6.5, 3.75), (6.5, 3.3))
arrow((6.5, 2.6), (6.5, 2.15))

arrow((5.4, 1.8), (4.6, 1.75))

arrow((3.65, 1.72), (2.6, 2.85), label="fail, retry_count < 1", curve=-0.2)
arrow((1.3, 3.3), (5.4, 2.95), label="retry\ngeneration", curve=-0.2)

arrow((4.6, 1.7), (9.4, 1.4), label="fail, retry_count >= 1  (safe-failure)", curve=0.25)

arrow((2.6, 2.05), (9.4, 3.3), label="pass", curve=0.35)

arrow((10.6, 0.9), (10.6, 3.1))

# Legend -------------------------------------------------------------------
legend_items = [
    mpatches.Patch(color=COLORS["deterministic"], label="Deterministic code node"),
    mpatches.Patch(color=COLORS["model"], label="Local Hugging Face model node"),
    mpatches.Patch(color=COLORS["decision"], label="Conditional routing"),
    mpatches.Patch(color=COLORS["terminal"], label="Terminal / success"),
    mpatches.Patch(color=COLORS["fail"], label="Terminal / safe failure"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.0, -0.02),
          fontsize=9, frameon=False, ncol=1)

ax.text(6.5, 7.85, "OrbitDesk Support Agent -- LangGraph Architecture",
        ha="center", fontsize=15, weight="bold")
ax.text(6.5, 0.15,
        "Loop protection: retry_count is capped at MAX_RETRIES=1, guaranteeing generation runs at most twice per request.",
        ha="center", fontsize=8.5, style="italic", color="#4b5563")

plt.tight_layout()
plt.savefig("diagram/graph_diagram.png", dpi=200, bbox_inches="tight")
print("Saved diagram/graph_diagram.png")
