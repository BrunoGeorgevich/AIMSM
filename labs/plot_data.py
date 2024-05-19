from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
from random import choice
import numpy as np
import os
import sys


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
    if len(sys.argv) > 1:
        DATABASE_NAME = sys.argv[1]
    else:
        print("Please provide the DATABASE_NAME as a command line argument.")
        sys.exit(1)
    # DATABASE_NAME = "all_model_experiment.csv"
    # DATABASE_NAME = "single_model_without_AIMSM.csv"
    # DATABASE_NAME = "run_individual_models_experiment.csv"
    # DATABASE_NAME = "random_switch_models_experiment.csv"
    # DATABASE_NAME = "context_switching_experiment_database.csv"

    SKIP_COLUMNS = ["Timestamp", "Running Models"]
    # SKIP_COLUMNS = ["FPS", "Running Models"]

    SKIP_GROUPS = ["None"]
    # SKIP_GROUPS = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
    if "all_model_experiment" in DATABASE_NAME:
        SKIP_GROUPS = [
            "None",
            "Yolo V8",
            "Yolo V8,Fast SAM",
            "Yolo V8,Fast SAM,Image Captioning",
            "Yolo V8,Fast SAM,Room Classification,Image Captioning",
        ]

    Y_PADDING = 2
    WS = 15

    df = pd.read_csv(os.path.join(DATABASE_NAME), sep=";")

    axis_labels = {
        "FPS": {
            "X": "elapsed time (s)",
            "Y": "",  # "ylim": (-1, 100),
            "ylim": (0, 20.1),
            "Title": "Frames per second (FPS)",
        },
        "CPU": {
            "X": "elapsed time (s)",
            "Y": "%",
            "ylim": (0, 20.1),
            "Title": "CPU usage",
        },
        "GPU": {
            "X": "elapsed time (s)",
            "Y": "%",
            "ylim": (0, 100.1),
            "Title": "GPU usage",
        },
        "RAM": {
            "X": "elapsed time (s)",
            "Y": "GB",
            "ylim": (0, 4.51),
            "Title": "RAM usage",
        },
        "VRAM": {
            "X": "elapsed time (s)",
            "Y": "GB",
            "ylim": (0, 2.01),
            "Title": "VRAM usage",
        },
    }

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    num_subplots = len(df.columns) - len(SKIP_COLUMNS)

    # SHARE_X = False
    # fig = plt.figure(
    #     layout="constrained",
    #     figsize=(18, 6),
    # )
    # gs = GridSpec(3, 2, figure=fig)
    # axs = [fig.add_subplot(gs[0, :])]
    # axs += [fig.add_subplot(gs[i, j]) for i in range(1, 3) for j in range(2)]

    SHARE_X = True

    if DATABASE_NAME == "all_model_experiment.csv":
        fig, axs = plt.subplots(num_subplots, figsize=(6, 6), sharex=SHARE_X, squeeze=True)
    else:
        fig, axs = plt.subplots(num_subplots, figsize=(4, 8), sharex=SHARE_X, squeeze=True)

    axs = axs.flatten()

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.35, wspace=0)

    # if "ylim" in axis_labels["FPS"]:
    #     df["FPS"] = df["FPS"].clip(upper=axis_labels["FPS"]["ylim"][1] - Y_PADDING)

    df["FPS"] = df["FPS"].rolling(WS).mean()
    df["CPU"] = df["CPU"].rolling(WS).mean()
    df["GPU"] = df["GPU"].rolling(WS).mean()
    df["RAM"] = df["RAM"].rolling(WS).mean()
    df["VRAM"] = df["VRAM"].rolling(WS).mean()

    x_axis = df["Timestamp"] - df["Timestamp"].min()

    print(len(x_axis), len(df["FPS"]))

    plot_idx = 0
    for column in df.columns:
        if column in SKIP_COLUMNS:
            continue

        axs[plot_idx].plot(x_axis, df[column], color=colors[plot_idx % len(colors)])
        axs[plot_idx].set_title(axis_labels[column]["Title"], fontweight='bold', fontsize=12)
        axs[plot_idx].set_ylabel(axis_labels[column]["Y"], fontsize=12)
        axs[plot_idx].tick_params(axis="both", which="major", labelsize=12)
        axs[plot_idx].yaxis.grid()

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

        pace = 10
        try:
            axs[plot_idx].set_xticks(np.arange(0, x_axis.max() + 1, pace))
        except ValueError:
            axs[plot_idx].set_xticks(np.arange(0, x_axis.max() + 1, 5))
        if SHARE_X:
            if plot_idx == num_subplots - 1:
                axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=12)
        else:
            axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=12)

        plot_idx += 1

    grouped = group_by_running_models(df)
    # grouped = sorted(grouped, key=lambda x: x[1].shape[0], reverse=True)


    colors = [
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
    # grouped = sorted(grouped, key=lambda x: x[1].shape[0])
    index = 0
    for name in groups_names:
        groups = grouped[name]
        for group in groups:
            if name not in group_colors:
                # group_colors[name] = generate_random_color()
                group_colors[name] = colors[index]
                index += 1

            plot_idx = 0
            for column in df.columns:
                if column in SKIP_COLUMNS:
                    continue

                start = x_axis[group.index.min()]
                end = x_axis[group.index.max()]

                if name not in SKIP_GROUPS:
                    axs[plot_idx].axvspan(start, end, color=group_colors[name], alpha=0.15)

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

                plot_idx += 1

    print(groups_names)
    print(created_legends)

    if len(legends) > 0:
        plt.legend(
            loc="center",
            bbox_to_anchor=(0.45, -0.6 - (0.055 * num_subplots), 0, 0),
            handles=legends,
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize="medium",
        )


    bottom_margin = 0.15 if len(legends) > 0 else 0.08
    plt.subplots_adjust(left=0.1, right=0.99, bottom=bottom_margin, top=0.95)

    base_name = os.path.splitext(os.path.basename(DATABASE_NAME))[0]
    plt.savefig(f"{base_name}.pdf", bbox_inches='tight', pad_inches=0.05)

    plt.show()
