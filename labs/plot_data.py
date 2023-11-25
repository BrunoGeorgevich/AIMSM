from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Ler o arquivo CSV
df = pd.read_csv(os.path.join("..", "database.csv"), sep=";")

# Cria uma figura e um array de eixos
fig, axs = plt.subplots(5)
# Ajusta o espaço entre os subplots
plt.subplots_adjust(hspace=1)

# Define um ciclo de cores
colors = ["b", "g", "r", "c", "m", "y", "k"]

axis_labels = {
    "FPS": {
        "X": "s",
        "Y": "",
        "ylim": (-10, 100),
        "Title": "Frames per second",
    },
    "CPU": {
        "X": "s",
        "Y": "%",
        "ylim": (0, 100),
        "Title": "CPU usage",
    },
    "RAM": {
        "X": "s",
        "Y": "GB",
        "ylim": (0, 10),
        "Title": "RAM usage",
    },
    "GPU": {
        "X": "s",
        "Y": "%",
        "ylim": (0, 100),
        "Title": "GPU usage",
    },
    "VRAM": {
        "X": "s",
        "Y": "GB",
        "ylim": (0, 24),
        "Title": "VRAM usage",
    },
}

ws = 2
y_padding = 10
df["FPS"] = df["FPS"].clip(upper=axis_labels["FPS"]["ylim"][1] - y_padding)
df["FPS"] = df["FPS"].rolling(ws).mean()
df["CPU"] = df["CPU"].rolling(ws).mean()
df["RAM"] = df["RAM"].rolling(ws).mean()
df["GPU"] = df["GPU"].rolling(ws).mean()
df["VRAM"] = df["VRAM"].rolling(ws).mean()

for i, column in enumerate(df.columns):
    if column == "Running Models":
        continue
    axs[i].plot(df[column], color=colors[i % len(colors)])
    axs[i].set_title(axis_labels[column]["Title"])
    axs[i].set_ylabel(axis_labels[column]["Y"])

    if "ylim" in axis_labels[column]:
        axs[i].set_ylim(axis_labels[column]["ylim"])

    if i == len(df.columns) - 1:
        axs[i].set_xlabel(axis_labels[column]["X"])

grouped = df.groupby("Running Models")

colors = ["b", "g", "r", "c", "m", "y", "k"]
legends = []
created_legends = []
for i, (name, group) in enumerate(grouped):
    if pd.isna(name):
        continue
    for j, column in enumerate(df.columns):
        if column == "Running Models":
            continue
        start = group.index.min()
        end = group.index.max()
        axs[j].axvspan(start, end, color=colors[i % len(colors)], alpha=0.15)

        parsed_name = name.replace(",", " + ")
        if parsed_name not in created_legends:
            legends.append(
                Patch(facecolor=colors[i % len(colors)], alpha=0.15, label=parsed_name)
            )
            created_legends.append(parsed_name)


legends = list(set(legends))
# Adiciona as legendas ao gráfico
plt.legend(handles=legends)


plt.show()
