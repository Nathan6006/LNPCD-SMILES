"""
Publication figure for the LNP cytotoxicity dataset (LNPCD).

Panel a  UMAP of Morgan count fingerprints (r=2, 1024 bit) for every unique
         ionizable lipid, coloured by source library.
Panel b  Composition of the dataset: helper lipid, cell line, cargo type.

Usage
-----
    python figure_2/make_dataset_overview.py

Writes lnpcd_dataset_overview.svg (vector, publication ready) and a 600 dpi PNG
of the same figure, next to this script. The fingerprint/UMAP step is cached in
.umap_cache.npz so re-styling the figure does not re-run the embedding.

Requires: rdkit, umap-learn, pandas, numpy, matplotlib.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "lnpcd.csv"
CACHE = HERE / ".umap_cache.npz"
OUT_SVG = HERE / "lnpcd_dataset_overview.svg"
OUT_PNG = HERE / "lnpcd_dataset_overview.png"

FP_RADIUS = 2
FP_BITS = 1024
SVD_COMPONENTS = 50

# Getting a *connected* embedding out of chemical fingerprints takes three
# deliberate choices, all of them fighting the same failure mode -- binary
# fingerprints under Tanimoto give a near-bimodal distance distribution (0 within
# a combinatorial sub-series, ~1 between them), which shatters the map into
# dozens of unrelated specks:
#   1. count fingerprints, log-damped and L2-normalised -> graded, not on/off;
#   2. a 50-component SVD -> dense, continuous coordinates with a real metric;
#   3. a large neighbourhood, large min_dist/spread and weak repulsion -> UMAP
#      preserves global layout instead of blowing every micro-cluster apart.
# A large min_dist/spread additionally puffs each cluster up, so the map reads as
# broad regions filling the frame rather than specks separated by empty space.
UMAP_KW = dict(
    n_neighbors=80,
    min_dist=0.99,
    spread=2.0,
    repulsion_strength=0.30,
    negative_sample_rate=5,
    n_epochs=1000,
    metric="cosine",
    n_components=2,
    random_state=0,
)

# Categorical palette, 11 slots, validated for CVD separation and normal-vision
# separation on adjacent pairs against a near-white surface.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#00a6c7",  # cyan
    "#a349c9",  # purple
    "#8a9b0f",  # olive
]

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#84837c"
SURFACE = "#ffffff"
RULE = "#d9d8d2"

# Human-readable library names, keyed by Experiment_ID (cell-line suffixes on
# the dendrimer study are dropped -- it is a single library screened twice).
LIBRARY_NAMES = {
    "Liu_iphos": "Liu — ionizable phospholipid",
    "Xue_CAD_LNP": "Xue — cationic amphiphilic drug",
    "Han_amidine": "Han — amidine",
    "Han_a3": "Han — A3 multicomponent",
    "Lee_unsat": "Lee — unsaturated",
    "Lin_peptide": "Lin — peptide",
    "Miller_zwitter": "Miller — zwitterionic (ZAL)",
    "Yu_Aminoglycoside": "Yu — aminoglycoside",
    "Han_branched": "Han — branched ester",
    "Zhang_Aminoglycoside": "Zhang — aminoglycoside",
    "Farbiak_Dendrimer": "Farbiak — dendrimer",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["library"] = (
        df["Experiment_ID"].str.replace("_HeLa", "", regex=False).str.replace("_igrov1", "", regex=False)
    )
    df["Helper_lipid_ID"] = df["Helper_lipid_ID"].fillna("None")
    return df


def embed(smiles: list[str]) -> np.ndarray:
    """Morgan count fingerprints -> SVD -> 2-D UMAP. Cached on disk."""
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        if list(cached["smiles"]) == smiles:
            return cached["xy"]

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.decomposition import TruncatedSVD
    import umap

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

    fps = np.zeros((len(smiles), FP_BITS), dtype=float)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES at row {i}: {smi}")
        fps[i] = gen.GetCountFingerprintAsNumPy(mol)

    fps = np.log1p(fps)  # damp highly repeated substructures (long alkyl tails)
    fps /= np.linalg.norm(fps, axis=1, keepdims=True)
    latent = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=0).fit_transform(fps)

    xy = np.asarray(umap.UMAP(**UMAP_KW).fit_transform(latent), dtype=float)
    np.savez(CACHE, smiles=np.array(smiles, dtype=object), xy=xy)
    return xy


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            # Arial is the Nature-family figure face, and what the reference uses.
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_letter(fig, x, y, letter, label):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", color=INK, va="baseline")
    fig.text(x + 0.018, y, label, fontsize=8.5, color=INK_2, va="baseline")


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def draw_umap(ax, lipids: pd.DataFrame, order: list[str], colors: dict) -> None:
    for lib in order:  # painted largest-first so small libraries stay visible
        sub = lipids[lipids["library"] == lib]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=9,
            c=colors[lib],
            linewidths=0.25,
            edgecolors=SURFACE,
            alpha=0.95,
        )

    ax.set_xlabel("UMAP 1", fontsize=7.5)
    ax.set_ylabel("UMAP 2", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.07)


R_OUTER, R_WIDTH = 1.0, 0.42


def draw_donut(ax, counts: pd.Series, title: str) -> None:
    """Donut sitting in the left half of its cell, with a colour legend beside it
    carrying every category name and share -- nothing is written on the ring."""
    total = counts.sum()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]

    ax.pie(
        counts.to_numpy(),
        colors=colors,
        startangle=90,
        counterclock=False,
        center=(0, 0),
        radius=R_OUTER,
        wedgeprops=dict(width=R_WIDTH, linewidth=1.6, edgecolor=SURFACE),
    )

    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=4.2,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=0.3,
            label=f"{name}   {100 * value / total:.1f}%",
        )
        for color, (name, value) in zip(colors, counts.items())
    ]
    legend = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.06, 0.5),
        bbox_transform=ax.transData,
        frameon=False,
        fontsize=6.4,
        handletextpad=0.45,
        labelspacing=0.62,
        borderpad=0,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)

    ax.text(0, 1.38, title, ha="center", va="center", fontsize=8, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 2.85)
    ax.set_ylim(-1.12, 1.55)
    ax.set_axis_off()


def rounded_panel(fig, x0, y0, x1, y1, label, subtitle) -> None:
    """The rounded card that groups panel b, with its label inside the corner."""
    fig.patches.append(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.016",
            transform=fig.transFigure,
            facecolor="none", edgecolor=RULE, linewidth=0.9,
            zorder=0,
        )
    )
    fig.text(x0 + 0.022, y1 - 0.030, label, fontsize=8, color=INK, va="center")
    fig.text(x0 + 0.022, y1 - 0.058, subtitle, fontsize=6.4, color=INK_MUTED, va="center")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def main() -> None:
    set_style()
    df = load_data()

    lipids = df.drop_duplicates("smiles").reset_index(drop=True)
    xy = embed(lipids["smiles"].tolist())
    lipids["x"], lipids["y"] = xy[:, 0], xy[:, 1]

    order = lipids["library"].value_counts().index.tolist()
    colors = {lib: PALETTE[i] for i, lib in enumerate(order)}

    fig = plt.figure(figsize=(7.2, 6.5))
    gs_a = fig.add_gridspec(
        1, 2, width_ratios=[1.95, 1.0],
        left=0.055, right=0.995, top=0.915, bottom=0.425, wspace=0.06,
    )
    gs_b = fig.add_gridspec(
        1, 3, left=0.052, right=0.985, top=0.245, bottom=0.042, wspace=0.10,
    )

    # -- panel a: UMAP + legend --------------------------------------------- #
    ax_umap = fig.add_subplot(gs_a[0, 0])
    draw_umap(ax_umap, lipids, order, colors)

    ax_leg = fig.add_subplot(gs_a[0, 1])
    ax_leg.set_axis_off()
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=4.5,
            markerfacecolor=colors[lib], markeredgecolor=SURFACE, markeredgewidth=0.3,
            label=f"{LIBRARY_NAMES[lib]}  ({(lipids['library'] == lib).sum()})",
        )
        for lib in order
    ]
    leg = ax_leg.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=6.2,
        handletextpad=0.5,
        labelspacing=0.72,
        borderpad=0,
        title="Source library (lipids)",
        title_fontsize=6.8,
    )
    leg.get_title().set_color(INK)
    leg.get_title().set_fontweight("bold")
    for text in leg.get_texts():
        text.set_color(INK_2)

    panel_letter(fig, 0.012, 0.960, "a", "Ionizable lipid chemical space")
    fig.text(
        0.055,
        0.936,
        f"UMAP of Morgan count fingerprints (r = {FP_RADIUS}, {FP_BITS} bit) — "
        f"{len(lipids):,} unique ionizable lipids from {len(order)} published libraries",
        fontsize=6.4,
        color=INK_MUTED,
    )

    # -- panel b: composition donuts ---------------------------------------- #
    specs = [
        (df["Helper_lipid_ID"].value_counts(), "Helper lipid"),
        (df["Model_type"].value_counts().rename({"MDA_MB": "MDA-MB-231"}), "Cell line"),
        (df["Cargo_type"].value_counts(), "Cargo type"),
    ]
    for col, (counts, title) in enumerate(specs):
        ax = fig.add_subplot(gs_b[0, col])
        draw_donut(ax, counts, title)

    fig.text(0.008, 0.352, "b", fontsize=11, fontweight="bold", color=INK, va="baseline")
    rounded_panel(
        fig, 0.030, 0.020, 0.995, 0.344,
        "Dataset composition",
        f"{len(df):,} viability measurements across "
        f"{lipids['library'].nunique()} published lipid libraries",
    )

    fig.savefig(OUT_SVG, format="svg")
    fig.savefig(OUT_PNG, dpi=600)
    print(f"wrote {OUT_SVG}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
