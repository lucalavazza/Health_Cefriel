from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pipeline.config import DISPLAY_LABELS


def display_label(variable_name: str) -> str:
    return DISPLAY_LABELS.get(variable_name, variable_name.replace("_", " "))


def display_labels(variable_names):
    return [display_label(name) for name in variable_names]


def graph_display_label(variable_name: str) -> str:
    label = display_label(variable_name)
    return label.replace(" ", "\n") if len(label) > 14 else label


def graph_display_labels(variable_names):
    return [graph_display_label(name) for name in variable_names]


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def apply_axis_style(ax, rotation: int = 30):
    ax.tick_params(axis="x", labelrotation=rotation)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    return ax


def finalize_figure(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
