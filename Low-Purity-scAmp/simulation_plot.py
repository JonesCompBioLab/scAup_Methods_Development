from pathlib import Path
import pandas as pd
import re

root = Path("./lps_results_v2_withnoamp/")

methods = {
    "scAmp": "NN_results.tsv",
    "GMM + KNN": "KNN_results.tsv",
    "GMM (Unscaled)": "GMM_results.tsv",
    "GMM (Scaled)": "GMM_scaled_results.tsv",
}

dfs = []

for folder in root.iterdir():
    if not folder.is_dir():
        continue

    # Parse parameters from folder name
    match = re.search(
        r"fmax_([^_]+)_overlap_([^_]+)_extracounts_([^_]+)_depth_([^_]+)",
        folder.name
    )

    if match is None:
        print(f"Skipping unrecognized folder: {folder.name}")
        continue

    fmax, overlap, extracounts, depth = map(float, match.groups())

    for method, filename in methods.items():
        filepath = folder / filename

        if not filepath.exists():
            print(f"Missing: {filepath}")
            continue

        df = pd.read_csv(filepath, sep="\t")

        # Keep / rename the columns we care about
        df = df[
            [
                "Num_species",
                "Comb_chance",
                "ecDNA_Percentage",
                "ecDNA",
                "runtime",
            ]
        ].copy()

        df["depth"] = depth
        df["overlap"] = overlap
        df["fmax"] = fmax
        df["extracounts"] = extracounts
        df["method"] = method

        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# Convenient lowercase names
all_data = all_data.rename(columns={
    "Num_species": "num_species",
    "Comb_chance": "comb_chance",
    "ecDNA_Percentage": "ecDNA_percentage",
})


# Individual dataframes if desired
scamp = all_data[all_data["method"] == "scAmp"].copy()
gmm_knn = all_data[all_data["method"] == "GMM + KNN"].copy()
gmm = all_data[all_data["method"] == "GMM (Unscaled)"].copy()
gmm_scaled = all_data[all_data["method"] == "GMM (Scaled)"].copy()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_data["Method"] = all_data["method"]



# ============================================================
# Plot 1: Performance vs. ecDNA percentage
# ============================================================

# 2.5% bins
bin_width = 2.5

all_data["ecDNA_bin"] = (
    np.floor(all_data["ecDNA_percentage"] / bin_width) * bin_width + 1.25
)

# Calculate fraction True and number of observations
binned = (
    all_data
    .groupby(["method", "Method", "ecDNA_bin"])
    .agg(
        yes=("ecDNA", "sum"),
        n=("ecDNA", "size"),
    )
    .reset_index()
)

binned["percentage_yes"] = 100 * binned["yes"] / binned["n"]

plt.figure(figsize=(12, 7))

ax = sns.lineplot(
    data=binned,
    x="ecDNA_bin",
    y="percentage_yes",
    hue="Method",
    hue_order=[
        "GMM (Unscaled)",
        "GMM (Scaled)",
        "scAmp",
        "KNN",
    ],
    marker="o",
    markersize=0,
    alpha=0.5,
    legend=False
)

# Add points sized by number of observations
sns.scatterplot(
    data=binned,
    x="ecDNA_bin",
    y="percentage_yes",
    hue="Method",
    hue_order=[
        "GMM (Unscaled)",
        "GMM (Scaled)",
        "scAmp",
        "GMM + KNN",
    ],
    size="n",
    sizes=(30, 250),
    ax=ax,
    alpha=0.5,
    legend="brief"
)

# Explicitly create/get legend
legend = ax.legend(
    title="Method",
    fontsize=14,
    title_fontsize=16
)

plt.xlabel("ecDNA percentage", fontsize=16)
plt.ylabel("% classified as ecDNA", fontsize=16)
plt.title("ecDNA Detection Methods vs. ecDNA percentage", fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(0, 100)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig('./plots/ecDNA_percent_sim.pdf')
plt.close()











# ============================================================
# Plot 2: Performance vs. ecDNA percentage and depth
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Bin ecDNA percentage
# ------------------------------------------------------------

ecDNA_bin_width = 20

all_data["ecDNA_bin"] = (
    np.floor(all_data["ecDNA_percentage"] / ecDNA_bin_width)
    * ecDNA_bin_width + 10
)

# ------------------------------------------------------------
# Define depth values
# ------------------------------------------------------------


depth_unique = sorted(all_data["depth"].unique())

# If depth values are things like 0.25, 0.5, 1.0,
# display them directly.
depth_labels = depth_unique

# ------------------------------------------------------------
# Methods / plot order
# ------------------------------------------------------------

method_order = [
    "GMM (Unscaled)",
    "GMM (Scaled)",
    "scAmp",
    "GMM + KNN",
]

# ------------------------------------------------------------
# Calculate binned values
# ------------------------------------------------------------

binned_depth = (
    all_data
    .groupby(["Method", "depth", "ecDNA_bin"])
    .agg(
        yes=("ecDNA", "sum"),
        n=("ecDNA", "size"),
    )
    .reset_index()
)

binned_depth["percentage_yes"] = (
    100 * binned_depth["yes"] / binned_depth["n"]
)

# ------------------------------------------------------------
# Unique x values
# ------------------------------------------------------------

x_unique = sorted(all_data["ecDNA_bin"].unique())

# ------------------------------------------------------------
# Build Z matrices and count matrices
# ------------------------------------------------------------

all_Z = []
all_counts = []

for method in method_order:

    method_data = binned_depth[
        binned_depth["Method"] == method
    ]

    Z = np.full(
        (len(depth_unique), len(x_unique)),
        np.nan
    )

    counts = np.zeros(
        (len(depth_unique), len(x_unique)),
        dtype=int
    )

    for _, row in method_data.iterrows():

        i = depth_unique.index(row["depth"])
        j = x_unique.index(row["ecDNA_bin"])

        Z[i, j] = row["percentage_yes"]
        counts[i, j] = row["n"]

    all_Z.append(Z)
    all_counts.append(counts)

# ------------------------------------------------------------
# Color scale
# ------------------------------------------------------------

valid_values = [
    Z[np.isfinite(Z)]
    for Z in all_Z
]

all_values = np.concatenate(valid_values)

# Allow some wiggle room in color
vmin = max(0, all_values.min() - 0.001)
vmax = min(100, all_values.max() + 0.001)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 13)
)

axes = axes.flatten()

for ax, Z, counts, title in zip(
    axes,
    all_Z,
    all_counts,
    method_order
):

    im = ax.imshow(
        Z,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold"
    )

    # X axis
    ax.set_xticks(range(len(x_unique)))
    ax.set_xticklabels(
        [f"{x:.0f}" for x in x_unique],
        fontsize=13
    )

    # Y axis
    ax.set_yticks(range(len(depth_unique)))
    ax.set_yticklabels(
        depth_labels,
        fontsize=13
    )

    # Add labels
    for i in range(len(depth_unique)):
        for j in range(len(x_unique)):

            if counts[i, j] > 0:

                # Percentage
                ax.text(
                    j - 0.28,
                    i + 0.25,
                    f"{Z[i, j]:.1f}",
                    ha="left",
                    va="top",
                    color="white",
                    fontsize=12,
                    fontweight="bold"
                )

                # Number of observations
                ax.text(
                    j + 0.28,
                    i - 0.25,
                    f"n={counts[i, j]}",
                    ha="right",
                    va="bottom",
                    color="white",
                    fontsize=9
                )

# ------------------------------------------------------------
# Common axis labels
# ------------------------------------------------------------

for ax in axes[2:]:
    ax.set_xlabel(
        "ecDNA percentage (Binned)",
        fontsize=17
    )

for ax in axes[::2]:
    ax.set_ylabel(
        "Insertions per Copy Number (Saturation)",
        fontsize=17
    )

# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

fig.subplots_adjust(
    right=0.88,
    wspace=0.30,   # horizontal separation
    hspace=0.35    # vertical separation
)

cbar_ax = fig.add_axes(
    [0.90, 0.15, 0.02, 0.70]
)

cbar = fig.colorbar(
    im,
    cax=cbar_ax
)

cbar.set_label(
    "% classified as ecDNA",
    fontsize=16
)

cbar.ax.tick_params(
    labelsize=13
)

# plt.show()
plt.savefig('./plots/depth_sim.pdf')
plt.close()

# ============================================================
# Plot 3 & 4: Performance vs. ecDNA percentage and
#              Cosegregation Strength / Overlap
# ============================================================

method_order = [
    "GMM (Unscaled)",
    "GMM (Scaled)",
    "scAmp",
    "GMM + KNN",
]


def plot_parameter_heatmap(
    data,
    parameter,
    parameter_label,
    output_file,
    figsize=(16, 13),
):
    """
    Plot % classified as ecDNA as a function of ecDNA percentage
    and a second simulation parameter (e.g. Comb_chance or overlap).
    """

    # --------------------------------------------------------
    # Bin ecDNA percentage
    # --------------------------------------------------------

    ecDNA_bin_width = 20

    data = data.copy()

    data["ecDNA_bin"] = (
        np.floor(data["ecDNA_percentage"] / ecDNA_bin_width)
        * ecDNA_bin_width
        + 10
    )

    # --------------------------------------------------------
    # Unique parameter values
    # --------------------------------------------------------

    parameter_unique = sorted(data[parameter].unique())

    parameter_labels = parameter_unique

    # --------------------------------------------------------
    # Calculate binned values
    # --------------------------------------------------------

    binned = (
        data
        .groupby(
            ["Method", parameter, "ecDNA_bin"]
        )
        .agg(
            yes=("ecDNA", "sum"),
            n=("ecDNA", "size"),
        )
        .reset_index()
    )

    binned["percentage_yes"] = (
        100 * binned["yes"] / binned["n"]
    )

    # --------------------------------------------------------
    # Unique x values
    # --------------------------------------------------------

    x_unique = sorted(data["ecDNA_bin"].unique())

    # --------------------------------------------------------
    # Build Z matrices and count matrices
    # --------------------------------------------------------

    all_Z = []
    all_counts = []

    for method in method_order:

        method_data = binned[
            binned["Method"] == method
        ]

        Z = np.full(
            (len(parameter_unique), len(x_unique)),
            np.nan
        )

        counts = np.zeros(
            (len(parameter_unique), len(x_unique)),
            dtype=int
        )

        for _, row in method_data.iterrows():

            i = parameter_unique.index(row[parameter])
            j = x_unique.index(row["ecDNA_bin"])

            Z[i, j] = row["percentage_yes"]
            counts[i, j] = row["n"]

        all_Z.append(Z)
        all_counts.append(counts)

    # --------------------------------------------------------
    # Color scale
    # --------------------------------------------------------

    valid_values = [
        Z[np.isfinite(Z)]
        for Z in all_Z
        if np.any(np.isfinite(Z))
    ]

    all_values = np.concatenate(valid_values)

    vmin = max(0, all_values.min() - 0.001)
    vmax = min(100, all_values.max() + 0.001)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize
    )

    axes = axes.flatten()

    for ax, Z, counts, title in zip(
        axes,
        all_Z,
        all_counts,
        method_order
    ):

        im = ax.imshow(
            Z,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto"
        )

        ax.set_title(
            title,
            fontsize=20,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        ax.set_xticks(range(len(x_unique)))
        ax.set_xticklabels(
            [f"{x:.0f}" for x in x_unique],
            fontsize=13
        )

        # ----------------------------------------------------
        # Y axis
        # ----------------------------------------------------

        ax.set_yticks(range(len(parameter_unique)))
        ax.set_yticklabels(
            parameter_labels,
            fontsize=13
        )

        # ----------------------------------------------------
        # Add values and n
        # ----------------------------------------------------

        for i in range(len(parameter_unique)):
            for j in range(len(x_unique)):

                if counts[i, j] > 0:

                    # Percentage
                    ax.text(
                        j - 0.28,
                        i + 0.25,
                        f"{Z[i, j]:.1f}",
                        ha="left",
                        va="top",
                        color="white",
                        fontsize=12,
                        fontweight="bold"
                    )

                    # Number of observations
                    ax.text(
                        j + 0.28,
                        i - 0.25,
                        f"n={counts[i, j]}",
                        ha="right",
                        va="bottom",
                        color="white",
                        fontsize=9
                    )

    # --------------------------------------------------------
    # Common axis labels
    # --------------------------------------------------------

    for ax in axes[2:]:
        ax.set_xlabel(
            "ecDNA percentage (Binned)",
            fontsize=17
        )

    for ax in axes[::2]:
        ax.set_ylabel(
            parameter_label,
            fontsize=17
        )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    fig.subplots_adjust(
        right=0.88,
        wspace=0.30,
        hspace=0.35
    )

    cbar_ax = fig.add_axes(
        [0.90, 0.15, 0.02, 0.70]
    )

    cbar = fig.colorbar(
        im,
        cax=cbar_ax
    )

    cbar.set_label(
        "% classified as ecDNA",
        fontsize=16
    )

    cbar.ax.tick_params(
        labelsize=13
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Plot 3: Performance vs. ecDNA percentage and
#         Cosegregation Strength
# ============================================================

plot_parameter_heatmap(
    data=all_data,
    parameter="comb_chance",
    parameter_label="Cosegregation Strength",
    output_file="./plots/cosegregation_strength_sim.pdf",
)


# ============================================================
# Plot 4: Performance vs. ecDNA percentage and Overlap
# ============================================================

plot_parameter_heatmap(
    data=all_data,
    parameter="overlap",
    parameter_label="Overlap",
    output_file="./plots/overlap_sim.pdf",
)


exit(0)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=all_data,
    x="Method",
    y="runtime",
)

plt.xlabel("")
plt.title("Average Runtime of Methods")
plt.ylabel("Runtime (s)")
plt.tight_layout()
# plt.show()
plt.savefig('./plots/runtime_sim.pdf')
plt.close()