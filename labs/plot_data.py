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
    groups = []
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
            groups.append((group_name, pd.DataFrame(group)))
            group_name = row["Running Models"]
            group = [row]

    groups.append((group_name, pd.DataFrame(group)))

    return groups


# DATABASE_NAME = "all_model_experiment.csv"
# DATABASE_NAME = "switching_models_database.csv"
# DATABASE_NAME = "random_switch_models_experiment.csv"
DATABASE_NAME = "context_switching_experiment_database.csv"
# DATABASE_NAME = "yolov8.csv"
# DATABASE_NAME = "fastsam.csv"
# DATABASE_NAME = "room_classification.csv"
# DATABASE_NAME = "image_captioning.csv"
# DATABASE_NAME = "alternate_time_window.csv"

SKIP_COLUMNS = ["Running Models"]
# SKIP_COLUMNS = ["FPS", "Running Models"]

SKIP_GROUPS = []
# SKIP_GROUPS = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
SKIP_GROUPS = ["Yolo V8,Fast SAM,Image Captioning,Room Classification"]

Y_PADDING = 10
WS = 3

df = pd.read_csv(os.path.join("..", "data", DATABASE_NAME), sep=";")

axis_labels = {
    "FPS": {
        "X": "iterations",
        "Y": "",  # "ylim": (-1, 100),
        "ylim": (0, 20.1),
        "Title": "Frames per second (FPS)",
    },
    "CPU": {
        "X": "iterations",
        "Y": "%",
        "ylim": (0, 101),
        "Title": "CPU usage",
    },
    "GPU": {
        "X": "iterations",
        "Y": "%",
        "ylim": (0, 101),
        "Title": "GPU usage",
    },
    "RAM": {
        "X": "iterations",
        "Y": "GB",
        "ylim": (0, 8.1),
        "Title": "RAM usage",
    },
    "VRAM": {
        "X": "iterations",
        "Y": "GB",
        "ylim": (0, 8.1),
        "Title": "VRAM usage",
    },
}

colors = ["b", "g", "r", "c", "m", "y", "k"]
# fig = plt.figure(
#     layout="constrained",
#     figsize=(18, 6),
# )
# gs = GridSpec(3, 2, figure=fig)

# axs = [fig.add_subplot(gs[0, :])]
# axs += [fig.add_subplot(gs[i, j]) for i in range(1, 3) for j in range(2)]

SHARE_X = True
num_subplots = len(df.columns) - len(SKIP_COLUMNS)
fig, axs = plt.subplots(num_subplots, figsize=(8, 12), sharex=SHARE_X, squeeze=False)
axs = axs.flatten()


plt.subplots_adjust(hspace=0.4)

if "ylim" in axis_labels["FPS"]:
    df["FPS"] = df["FPS"].clip(upper=axis_labels["FPS"]["ylim"][1] - Y_PADDING)
df["FPS"] = df["FPS"].rolling(WS).mean()
df["CPU"] = df["CPU"].rolling(WS).mean()
df["GPU"] = df["GPU"].rolling(WS).mean()
df["RAM"] = df["RAM"].rolling(WS).mean()
df["VRAM"] = df["VRAM"].rolling(WS).mean()

plot_idx = 0
for column in df.columns:
    if column in SKIP_COLUMNS:
        continue

    axs[plot_idx].plot(df[column], color=colors[plot_idx % len(colors)])
    axs[plot_idx].set_title(axis_labels[column]["Title"], fontsize=18)
    axs[plot_idx].set_ylabel(axis_labels[column]["Y"], fontsize=14)
    axs[plot_idx].tick_params(axis="both", which="major", labelsize=14)
    axs[plot_idx].yaxis.grid()

    if "ylim" in axis_labels[column]:
        ylims = axis_labels[column]["ylim"]
        ticks = int(ylims[1] / 5)

        if axis_labels[column]["Y"] == "%":
            ticks = 20

        if axis_labels[column]["Y"] == "GB":
            ticks = 2

        axs[plot_idx].set_ylim(ylims)
        axs[plot_idx].yaxis.set(ticks=np.arange(ylims[0], ylims[1], ticks))

    if SHARE_X:
        if plot_idx == num_subplots - 1:
            axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=14)
    else:
        axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=14)

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

# grouped = sorted(grouped, key=lambda x: x[1].shape[0])

index = 0
for name, group in grouped:
    if pd.isna(name):
        continue

    if name not in group_colors:
        # group_colors[name] = generate_random_color()
        group_colors[name] = colors[index]
        index += 1

    plot_idx = 0
    for column in df.columns:
        if column in SKIP_COLUMNS:
            continue

        start = group.index.min()
        end = group.index.max()

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

if len(legends) > 0:
    plt.legend(
        loc="center",
        bbox_to_anchor=(0.5, -0.6 - (0.06 * num_subplots), 0, 0),
        handles=legends,
        ncol=2,
        fancybox=True,
        shadow=True,
        fontsize="large",
    )


bottom_margin = 0.15 if len(legends) > 0 else 0.08
plt.subplots_adjust(left=0.1, right=0.99, bottom=bottom_margin, top=0.95)

base_name = os.path.splitext(os.path.basename(DATABASE_NAME))[0]
plt.savefig(f"{base_name}.png")

plt.show()
