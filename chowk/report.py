"""Charts for the README, drawn from artifacts/reports/*.json so they always match the numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chowk import config

COLORS = {"truth": "#4a4a4a", "chowk": "#1b7f5f", "zero_shot": "#c8553d", "naive": "#9a9a9a"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def session_pcu_chart(replay_rows: list[dict], zero_shot_pcu: dict[int, float], out: Path) -> None:
    plt = _plt()
    rows = [r for r in replay_rows if r["vehicles"] >= 20]
    rows.sort(key=lambda r: r["pcu_true"])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(x - 0.27, [r["pcu_true"] for r in rows], 0.27, label="True PCU (hand labels)", color=COLORS["truth"])
    ax.bar(x, [r["pcu_predicted"] for r in rows], 0.27, label="Chowk", color=COLORS["chowk"])
    ax.bar(x + 0.27, [zero_shot_pcu[r["session"]] for r in rows], 0.27, label="Off-the-shelf detector",
           color=COLORS["zero_shot"])
    ax.set_xticks(x, [str(r["session"]) for r in rows], fontsize=7)
    ax.set_xlabel("Recording session (one street clip), sorted by true PCU")
    ax.set_ylabel("Total PCU in the session")
    ax.legend(frameon=False, ncols=3, loc="upper left")
    ax.set_title("Per-session PCU: held-out predictions vs truth")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def confusion_chart(confusions: dict[str, list[list[int]]], out: Path) -> None:
    plt = _plt()
    names = ["two-wheeler", "light", "medium", "heavy"]
    fig, axes = plt.subplots(1, len(confusions), figsize=(4.2 * len(confusions), 3.8))
    for ax, (title, cm) in zip(np.atleast_1d(axes), confusions.items(), strict=True):
        cm = np.array(cm, dtype=float)
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Greens", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if frac[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(4), names, rotation=30)
        ax.set_yticks(range(4), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (hand label)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def robustness_chart(robust: dict, clean_err: float, out: Path) -> None:
    plt = _plt()
    names = list(robust)
    x = np.arange(len(names))
    series = [("trained_clean", "Trained on clean crops", COLORS["zero_shot"]),
              ("trained_without_this", "Trained with the other 3 degradations", COLORS["naive"]),
              ("trained_augmented", "Trained with this degradation too", COLORS["chowk"])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, (key, label, color) in enumerate(series):
        vals = [robust[n][key]["pcu"]["median_session_error"] * 100 for n in names]
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=label, color=color)
    ax.axhline(clean_err * 100, color=COLORS["truth"], lw=1, ls="--")
    ax.text(len(names) - 0.5, clean_err * 100 + 2, f"clean images: {clean_err:.1%}", ha="right", fontsize=8)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylabel("Median session PCU error (%)")
    ax.set_title("When the camera image gets worse (held-out sessions)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def example_sheet(archive, df, pred, out: Path, n: int = 8) -> None:
    """A strip of real crops per class with the prediction, for the README."""
    from PIL import Image, ImageDraw

    from chowk import data

    w = 120
    sheet = Image.new("RGB", (w * n, w * len(config.CLASSES) + 4), "white")
    draw = ImageDraw.Draw(sheet)
    rng = np.random.default_rng(config.SEED)
    for r, cls in enumerate(config.CLASSES):
        idx = np.nonzero((df["label"] == cls).to_numpy())[0]
        for c, i in enumerate(rng.choice(idx, n, replace=False)):
            im = data.load_image(archive, df["path"].iat[i])
            im.thumbnail((w - 6, w - 22))
            sheet.paste(im, (c * w + 3, r * w + 3))
            ok = pred[i] == cls
            draw.text((c * w + 4, r * w + w - 18), f"{cls if c == 0 else ''}{'' if ok else ' -> ' + pred[i]}",
                      fill="black" if ok else "red")
    sheet.save(out)
