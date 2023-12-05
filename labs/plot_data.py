from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
from random import choice
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
# DATABASE_NAME = "switching_models_database_v3.csv"
# DATABASE_NAME = "yolov8.csv"
# DATABASE_NAME = "fastsam.csv"
# DATABASE_NAME = "room_classification.csv"
# DATABASE_NAME = "image_captioning.csv"
# DATABASE_NAME = "alternate_time_window.csv"
# DATABASE_NAME = "random_switch_models_experiment.csv"
DATABASE_NAME = "context_switching_experiment_database.csv"

SKIP_COLUMNS = ["Running Models"]
SKIP_COLUMNS = ["FPS", "Running Models"]

SKIP_GROUPS = []
# SKIP_GROUPS = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
# SKIP_GROUPS = ["Yolo V8,Fast SAM,Image Captioning,Room Classification"]

WIDE_FIGURE = True
Y_PADDING = 10
WS = 3

df = pd.read_csv(os.path.join("..", "data", DATABASE_NAME), sep=";")

axis_labels = {
    "FPS": {
        "X": "s",
        "Y": "",  # "ylim": (-1, 100),
        "ylim": (-1, 30),
        "Title": "Frames per second",
    },
    "CPU": {
        "X": "s",
        "Y": "%",
        "ylim": (-1, 100),
        "Title": "CPU usage",
    },
    "GPU": {
        "X": "s",
        "Y": "%",
        "ylim": (-1, 100),
        "Title": "GPU usage",
    },
    "RAM": {
        "X": "s",
        "Y": "GB",
        "ylim": (-1, 10),
        "Title": "RAM usage",
    },
    "VRAM": {
        "X": "s",
        "Y": "GB",
        "ylim": (-1, 10),
        "Title": "VRAM usage",
    },
}

colors = ["b", "g", "r", "c", "m", "y", "k"]
num_subplots = len(df.columns) - len(SKIP_COLUMNS)
fig, axs = plt.subplots(
    num_subplots, figsize=(16 if not WIDE_FIGURE else 18, 9), sharex=True
)

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
    axs[plot_idx].tick_params(axis="both", which="major", labelsize=12)

    if "ylim" in axis_labels[column]:
        axs[plot_idx].set_ylim(axis_labels[column]["ylim"])

    if plot_idx == num_subplots - 1:
        axs[plot_idx].set_xlabel(axis_labels[column]["X"], fontsize=16)

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
        bbox_to_anchor=(0.5, -0.6 - (0.02 * num_subplots), 0, 0),
        handles=legends,
        ncol=5,
        fancybox=True,
        shadow=True,
        fontsize="x-large",
    )


bottom_margin = 0.15 if len(legends) > 0 else 0.08
plt.subplots_adjust(left=0.05, right=0.99, bottom=bottom_margin, top=0.95)

base_name = os.path.splitext(os.path.basename(DATABASE_NAME))[0]
plt.savefig(f"{base_name}.png")

plt.show()
