import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from constants import PATH_FIGURES


def tag_real_data_rows(df):
    df = df.copy()
    base_comparisons = ["Train vs Train", "Train vs Val", "Val vs Train"]
    df.loc[df["Comparison"].isin(base_comparisons), "Method"] = r"\method{Real Data}"
    return df


def replace_names(df):
    # Replace dataset names in the correct level
    dataset_patterns = {
        r"^abide_.*": r"\\dataset{ABIDE}",
        r"^adni_.*": r"\\dataset{ADNI}",
        r"^oasis3_.*": r"\\dataset{OASIS-3}",
        r"^bnci2014_002.*": r"\\dataset{BNCI}\\\\ \\dataset{2014-002}",
        r"^bnci2015_001.*": r"\\dataset{BNCI}\\\\ \\dataset{2015-001}",
    }

    for pattern, replacement in dataset_patterns.items():
        df["Dataset"] = df["Dataset"].str.replace(pattern, replacement, regex=True)

    # Replace Method names with LaTeX-friendly versions
    method_names = {
        "logeuclidean_DiffeoGauss": r"\method{DiffeoGauss}",
        "corrcholesky_DiffeoGauss": r"\method{DiffeoGauss}",
        "logeuclidean_DiffeoGauss_projected": r"\method{DiffeoGauss}",
        "corrcholesky_DiffeoGauss_projected": r"\method{DiffeoGauss}",
        "lower_triangular_DiffeoCFM_projected": r"\method{TriangCFM}",
        "lower_triangular_DiffeoCFM": r"\method{TriangCFM} (no proj.)",
        "strict_lower_triangular_DiffeoCFM_projected": r"\method{TriangCFM}",
        "strict_lower_triangular_DiffeoCFM": r"\method{TriangCFM} (no proj.)",
        "logeuclidean_DiffeoCFM": r"\proposed",
        "corrcholesky_DiffeoCFM": r"\proposed",
        "logeuclidean_DiffeoCFM_projected": r"\proposed",
        "corrcholesky_DiffeoCFM_projected": r"\proposed",
        "None_SPDConditionalFlowMatching": r"\method{RiemCFM}",
    }
    df["Method"] = df["Method"].replace(method_names)

    return df


def set_neurips_style():
    """Sets matplotlib rcParams for a NeurIPS-style figure."""
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.figsize": (7, 5.5),
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_process_data():
    """
    Loads metrics, calculates averaged F1 for generated and real data,
    and returns aggregated results and baselines.
    """
    all_gen_data = []
    all_real_data = []

    for modality in ["fmri", "eeg"]:
        path_modality = PATH_FIGURES / modality
        path_quality = path_modality / "quality_metrics.csv"
        path_gan_train = path_modality / "gan_train_metrics.csv"

        if not (path_quality.exists() and path_gan_train.exists()):
            print(f"Skipping modality '{modality}': CSV files not found.")
            continue

        df_quality = tag_real_data_rows(pd.read_csv(path_quality))
        df_gan_train = tag_real_data_rows(pd.read_csv(path_gan_train))

        # Calculate Alpha-Beta F1
        a = df_quality[r"$\alpha$-precision"]
        b = df_quality[r"$\beta$-recall"]
        df_quality[r"$\alpha$,$\beta$-F1"] = 2 * a * b / (a + b + 1e-12)

        # Process Generated Data
        df_quality_gen = df_quality[df_quality["Comparison"] == "Val vs Gen."].copy()
        df_gan_train_gen = df_gan_train[
            df_gan_train["Comparison"] == "Gen vs Val"
        ].copy()

        id_cols = ["Dataset", "Method", "Group", "Split"]
        quality_cols = id_cols + [r"$\alpha$,$\beta$-F1"]
        gan_cols = id_cols + ["F1", "Train time (s)", "Sampling time (s)"]

        df_merged_gen = pd.merge(
            df_quality_gen[quality_cols], df_gan_train_gen[gan_cols], on=id_cols
        )
        df_merged_gen["Avg F1"] = (
            df_merged_gen[r"$\alpha$,$\beta$-F1"] + df_merged_gen["F1"]
        ) / 2
        df_merged_gen["modality"] = modality
        all_gen_data.append(df_merged_gen)

        # Process Real Data Baseline
        df_quality_real = df_quality[df_quality["Comparison"] == "Train vs Val"].copy()
        df_gan_train_real = df_gan_train[
            df_gan_train["Comparison"] == "Train vs Val"
        ].copy()

        df_merged_real = pd.merge(
            df_quality_real[quality_cols], df_gan_train_real[gan_cols], on=id_cols
        )
        df_merged_real["Avg F1"] = (
            df_merged_real[r"$\alpha$,$\beta$-F1"] + df_merged_real["F1"]
        ) / 2
        df_merged_real["modality"] = modality
        all_real_data.append(df_merged_real)

    if not all_gen_data:
        print("No generated data found to process. Exiting.")
        return None, None

    # Aggregate Generated Data
    final_gen_df = pd.concat(all_gen_data, ignore_index=True)
    final_gen_df = replace_names(final_gen_df)
    grouped = final_gen_df.groupby(["modality", "Method"])
    mean_df = grouped.mean(numeric_only=True)
    std_df = grouped.std(numeric_only=True).fillna(0)
    aggregated_results = mean_df.merge(
        std_df, on=["modality", "Method"], suffixes=("_mean", "_std")
    ).reset_index()

    # Aggregate Real Data Baseline
    final_real_df = pd.concat(all_real_data, ignore_index=True)
    baseline_f1 = final_real_df.groupby("modality")["Avg F1"].mean().to_dict()

    return aggregated_results, baseline_f1


def create_summary_figure(df, baseline_f1):
    """Generates a 2x2 summary figure of F1 vs. Time with NeurIPS styling."""
    if df is None or df.empty:
        print("Cannot create figure: No aggregated data.")
        return

    set_neurips_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex="col", sharey=True)

    methods_ordered = [
        r"\method{DiffeoGauss}",
        r"\method{TriangCFM}",
        r"\method{RiemCFM}",
        r"\proposed",
    ]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    markers = ["o", "s", "^", "D", "*"]
    method_styles = {
        method: {"color": colors[i % len(colors)], "marker": markers[i % len(markers)]}
        for i, method in enumerate(methods_ordered)
    }

    modalities = {
        "fmri": ("Correlation Matrices (fMRI)", axes[0]),
        "eeg": ("Covariance Matrices (EEG)", axes[1]),
    }

    def plot_point(ax, x_mean, y_mean, x_std, y_std, style, label):
        """Helper to plot a point with marker, shaded area, and error bars."""
        ax.plot(
            x_mean,
            y_mean,
            marker=style["marker"],
            color=style["color"],
            label=label,
            zorder=3,
        )
        ax.fill_between(
            [x_mean - x_std, x_mean + x_std],
            y_mean - y_std,
            y_mean + y_std,
            color=style["color"],
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )
        ax.errorbar(
            x=x_mean,
            y=y_mean,
            xerr=x_std,
            yerr=y_std,
            fmt="none",
            capsize=3,
            color=style["color"],
            elinewidth=1,
            zorder=2,
        )

    for modality, (title, (ax_train, ax_sample)) in modalities.items():
        sub_df = df[df["modality"] == modality]

        # Plot baseline
        line_style = {"color": "gray", "linestyle": "--", "linewidth": 1.5, "zorder": 0}
        ax_train.axhline(y=baseline_f1.get(modality, 0), **line_style)
        ax_sample.axhline(y=baseline_f1.get(modality, 0), **line_style)

        for method in methods_ordered:
            style = method_styles.get(method)
            method_data = sub_df[sub_df["Method"] == method]
            if method_data.empty:
                continue

            f1_mean, f1_std = (
                method_data["Avg F1_mean"].iloc[0],
                method_data["Avg F1_std"].iloc[0],
            )
            train_t_mean, train_t_std = (
                method_data["Train time (s)_mean"].iloc[0],
                method_data["Train time (s)_std"].iloc[0],
            )
            sample_t_mean, sample_t_std = (
                method_data["Sampling time (s)_mean"].iloc[0],
                method_data["Sampling time (s)_std"].iloc[0],
            )

            label_name = method.replace(r"\method{", "").replace("}", "")
            if label_name == r"\proposed":
                label_name = "DiffeoCFM"

            plot_point(
                ax_train, train_t_mean, f1_mean, train_t_std, f1_std, style, label_name
            )
            plot_point(
                ax_sample, sample_t_mean, f1_mean, sample_t_std, f1_std, style, None
            )

        ax_sample.yaxis.set_label_position("right")
        ax_sample.set_ylabel(title, rotation=-90, labelpad=20)

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

    axes[0, 0].set_ylabel("Average F1 Score")
    axes[1, 0].set_ylabel("Average F1 Score")
    axes[1, 0].set_xlabel("Training Time (s)")
    axes[1, 1].set_xlabel("Sampling Time (s)")

    # Legend (in two rows)
    handles, labels = axes[1, 0].get_legend_handles_labels()

    baseline_handle = Line2D([0], [0], color="gray", linestyle="--", label="Real Data")
    handles.append(baseline_handle)
    labels.append("Real Data")

    # Set ncol to 3 to create two rows of three items
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
    )

    # Adjust layout to make space for the taller legend
    fig.tight_layout(rect=[0, 0, 0.95, 0.94])

    output_path = PATH_FIGURES / "f1_vs_time_summary.svg"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Figure saved to {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    aggregated_data, baseline_data = load_and_process_data()
    create_summary_figure(aggregated_data, baseline_data)
