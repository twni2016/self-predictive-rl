import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from absl import app
from absl import flags

if __name__ == "__main__":
    flags.DEFINE_string(
        "results_path", "./results/mountaincar.pkl", "Path to the saved results."
    )
    flags.DEFINE_string("output_path", None, "Filename for the saved figure.")
    flags.DEFINE_string("figure_title", None, "Title to add to the figure.")
    flags.DEFINE_boolean("with_legend", False, "If flag present, add a legend.")

FLAGS = flags.FLAGS

sns.set_style("whitegrid", {"grid.linestyle": "--"})
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["axes.grid"] = True
plt.rcParams["legend.loc"] = "best"
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.offset_threshold"] = 1
# plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True


def cosine_similarity(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.dot(x, y)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with open(FLAGS.results_path, "rb") as f:
        results = pickle.load(f)

    X_AXIS_LABEL = "Iterations"
    Y_AXIS_LABEL = " abs. cosine similarity"

    for log in results:
        params = log.pop("params")
        log[Y_AXIS_LABEL] = np.abs(cosine_similarity(params[:, 0], params[:, 1]))
        log["Iterations"] = log["step"]

    data = pd.DataFrame.from_records(results)

    ax = sns.lineplot(
        data=data,
        x=X_AXIS_LABEL,
        y=Y_AXIS_LABEL,
        units="seed",
        hue="use_stop_gradient",
        estimator=None,  # show all seeds
        legend=False,
        alpha=0.2,
    )

    ax = sns.lineplot(
        data=data,
        x=X_AXIS_LABEL,
        y=Y_AXIS_LABEL,
        hue="use_stop_gradient",
        estimator=np.median,
        ax=ax,
        linewidth=2.0,
        palette=sns.color_palette("dark", 3, desat=0.9),
    )
    plt.yscale("log")
    plt.yticks([10 ** (2 * i - 8) for i in range(4)])
    plt.xticks(np.arange(0, 501, step=100))
    plt.ylim(1e-10, 1)

    if FLAGS.with_legend:
        ax.legend(framealpha=0.2)  # must use the returned ans
    else:
        ax.legend().set_visible(False)

    if FLAGS.figure_title:
        plt.title(FLAGS.figure_title)

    filename = FLAGS.output_path
    if filename is None:
        filename = os.path.splitext(FLAGS.results_path)[0] + ".pdf"
    plt.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    app.run(main)
