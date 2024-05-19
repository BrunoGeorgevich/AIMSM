from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
from random import choice
import numpy as np
import os


def generate_random_color():
    return "#" + "".join([choice("0123456789ABCDEF") for _ in range(6)])


def group_by_running_models(df):
    groups = {}
    group = []
    group_name = None

    for _, row in df.iterrows():
        if group_name is None:
            group_name = row["Running Models"]
            group.append(row)
            continue

        if group_name == row["Running Models"]:
            group.append(row)
            continue
        else:
            if group_name is np.nan:
                group_name = "None"
            if group_name not in groups:
                groups[group_name] = [pd.DataFrame(group)]
            else:
                groups[group_name].append(pd.DataFrame(group))
            group_name = row["Running Models"]
            group = [row]

    if group_name is np.nan:
        group_name = "None"
    if group_name not in groups:
        groups[group_name] = [pd.DataFrame(group)]
    else:
        groups[group_name].append(pd.DataFrame(group))
    del groups["None"]

    return groups


if __name__ == "__main__":
    # DATABASE_NAME = "all_model_experiment.csv"
    # DATABASE_NAMES = [
    #     "all_model_experiment.csv",
    #     "single_model_without_AIMSM.csv",
    #     "run_individual_models_experiment.csv",
    # ]
    DATABASE_NAMES = [
        "random_switch_models_experiment.csv",
        "context_switching_experiment_database.csv",
    ]
    # DATABASE_NAME = "random_switch_models_experiment.csv"
    # DATABASE_NAME = "context_switching_experiment_database.csv"

    SKIP_COLUMNS = ["Timestamp", "Running Models"]
    # SKIP_COLUMNS = ["FPS", "Running Models"]

    SKIP_GROUPS = ["None"]
    # SKIP_GROUPS = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
    # if "all_model_experiment" in DATABASE_NAME:
    #     SKIP_GROUPS = [
    #         "None",
    #         "Yolo V8",
    #         "Yolo V8,Fast SAM",
    #         "Yolo V8,Fast SAM,Image Captioning",
    #         "Yolo V8,Fast SAM,Room Classification,Image Captioning",
    #     ]

    Y_PADDING = 2
    WS = 45

    axis_labels = {
        "FPS": {
            "X": "elapsed time (s)",
            "Y": "",  # "ylim": (-1, 100),
            "ylim": (0, 20.1),
            "Title": "FPS rate",
        },
        "CPU": {
            "X": "elapsed time (s)",
            "Y": "%",
            "ylim": (0, 17),
            "Title": "CPU usage",
        },
        "GPU": {
            "X": "elapsed time (s)",
            "Y": "%",
            "ylim": (0, 110),
            "Title": "GPU usage",
        },
        "RAM": {
            "X": "elapsed time (s)",
            "Y": "GB",
            "ylim": (0, 4.6),
            "Title": "RAM usage",
        },
        "VRAM": {
            "X": "elapsed time (s)",
            "Y": "GB",
            "ylim": (0, 2.3),
            "Title": "VRAM usage",
        },
    }

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    num_subplots = 10
    SHARE_X = True
    fig = plt.figure(
        layout="constrained",
        figsize=(8, 8),
    )
    gs = GridSpec(5, 2, figure=fig)
    axs = [fig.add_subplot(gs[i, j]) for j in range(2) for i in range(0, 5)]

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.07, wspace=0.03)

    # if "ylim" in axis_labels["FPS"]:
    #     df["FPS"] = df["FPS"].clip(upper=axis_labels["FPS"]["ylim"][1] - Y_PADDING)

    # df["FPS"] = df["FPS"].rolling(WS).mean()
    # df["CPU"] = df["CPU"].rolling(WS).mean()
    # df["GPU"] = df["GPU"].rolling(WS).mean()
    # df["RAM"] = df["RAM"].rolling(WS).mean()
    # df["VRAM"] = df["VRAM"].rolling(WS).mean()

    references = [
        "(a) Random Switching Models",
        "(b) Context Switching Models",
        "(c) With AIMSM",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
        "(i)",
        "(j)",
    ]
    plot_idx = 0
    shift = 0
    for DATABASE_NAME in DATABASE_NAMES:
        df = pd.read_csv(os.path.join(DATABASE_NAME), sep=";")
        df["FPS"] = df["FPS"].rolling(WS).mean()
        df["CPU"] = df["CPU"].rolling(WS).mean()
        df["GPU"] = df["GPU"].rolling(WS).mean()
        df["RAM"] = df["RAM"].rolling(WS).mean()
        df["VRAM"] = df["VRAM"].rolling(WS).mean()
        x_axis = df["Timestamp"] - df["Timestamp"].min()
        for column in df.columns:
            if column in SKIP_COLUMNS:
                continue

            axs[plot_idx].yaxis.grid()

            c = colors[int(plot_idx % (num_subplots / 2)) % len(colors)]
            axs[plot_idx].axhline(df[column].mean(), linestyle="dotted", color=c)
            axs[plot_idx].plot(x_axis, df[column], color=c)

            if plot_idx >= (num_subplots / 2):
                axs[plot_idx].set_title(
                    axis_labels[column]["Title"], fontweight="bold", fontsize=12, loc="left", x=1.03, y=0.35
                )
            if plot_idx < (num_subplots / 2):
                axs[plot_idx].set_ylabel(axis_labels[column]["Y"], fontsize=12)
                axs[plot_idx].tick_params(axis="both", which="major", labelsize=12)
            else:
                axs[plot_idx].set_yticklabels([])

            if (plot_idx + 1) % (num_subplots / 2) == 0:
                axs[plot_idx].tick_params(axis="x", which="major", labelsize=12)
            else:
                axs[plot_idx].set_xticklabels([])

            if "ylim" in axis_labels[column]:
                ylims = axis_labels[column]["ylim"]
                ticks = int(ylims[1] / 5)

                if axis_labels[column]["Y"] == "%":
                    if axis_labels[column]["Title"] == "CPU usage":
                        ticks = 5
                        axs[plot_idx].yaxis.labelpad = 4
                    elif axis_labels[column]["Title"] == "GPU usage":
                        ticks = 20
                        axs[plot_idx].yaxis.labelpad = -4

                if axis_labels[column]["Y"] == "GB":
                    ticks = 1
                    axs[plot_idx].yaxis.labelpad = 8

                axs[plot_idx].set_ylim(ylims)
                axs[plot_idx].yaxis.set(ticks=np.arange(ylims[0], ylims[1], ticks))

            pace = 20
            try:
                axs[plot_idx].set_xticks(np.arange(0, x_axis.max() + 1, pace))
            except ValueError:
                axs[plot_idx].set_xticks(np.arange(0, x_axis.max() + 1, 10))
            if SHARE_X:
                if (
                    plot_idx % (num_subplots / 2) == num_subplots / 2 - 1
                    and plot_idx != 0
                ):
                    ref_text = references[int(plot_idx // (num_subplots / 2))]
                    axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=12)
                    axs[plot_idx].text(
                        0.5,
                        -0.7,
                        ref_text,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                        transform=axs[plot_idx].transAxes,
                    )
                    # axs[plot_idx].text(0.5, -1, '(b)', ha='center', va='bottom', transform=axs[plot_idx].transAxes)
            else:
                axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=12)

            plot_idx += 1

        if DATABASE_NAME == "all_model_experiment.csv":
            shift += 1
            continue

        grouped = group_by_running_models(df)

        gcolors = [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#CC8800",
            "#0088CC",
            "#8800CC",
        ]
        group_colors = {}
        legends = []
        created_legends = []

        groups_names = grouped.keys()
        groups_names = sorted(groups_names)
        index = int(shift * num_subplots / 2)
        for name in groups_names:
            groups = grouped[name]
            for group in groups:
                if name not in group_colors:
                    # group_colors[name] = generate_random_color()
                    group_colors[name] = gcolors[int(index % (num_subplots / 2))]
                    index += 1

                gplot_idx = int(shift * num_subplots / 2)
                for column in df.columns:
                    if column in SKIP_COLUMNS:
                        continue

                    start = x_axis[group.index.min()]
                    end = x_axis[group.index.max()]

                    if name not in SKIP_GROUPS:
                        axs[gplot_idx].axvspan(
                            start, end, color=group_colors[name], alpha=0.15
                        )

                    parsed_name = name.replace(",", " + ")
                    if parsed_name not in created_legends:
                        if name in SKIP_GROUPS:
                            continue
                        legends.append(
                            Patch(
                                facecolor=group_colors[name],
                                alpha=0.15,
                                label=parsed_name,
                            )
                        )
                        created_legends.append(parsed_name)

                    gplot_idx += 1

        if len(legends) > 0:
            axs[int(4 + (5 * shift))].legend(
                loc="center",
                bbox_to_anchor=(0.5, -1.3, 0, 0),
                handles=legends,
                ncol=1,
                fancybox=True,
                shadow=True,
                fontsize="large",
            )
        shift += 1

    bottom_margin = 0.15 if len(legends) > 0 else 0.08
    plt.subplots_adjust(left=0.1, right=0.99, bottom=bottom_margin, top=0.95)

    base_name = os.path.splitext(os.path.basename("combined_doubled.pdf"))[0]
    plt.savefig(f"{base_name}.pdf", bbox_inches="tight", pad_inches=0.05)

    plt.show()
