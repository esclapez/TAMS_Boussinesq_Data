"""Plot sampling of the basin boundary in latent space."""

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from podscore import PODScore
from podlatent_visualizer import PODScoreVisualizer

if __name__ == "__main__":
    dataset = "Naive"

    color_str = np.array(["k", "r"])
    marker_str = np.array(["x", "o"])
    colors = cm.magma(np.linspace(0.1, 0.9, 3))
    colors = np.roll(colors, 2, axis=0)

    samples = np.load(f"./{dataset}/Basin_PODspace_0p025.npy")
    edge = np.load("EdgeState.npy")

    pod_data_file = "POD_score_database.nc"
    score_builder = PODScore(40 + 1, 80 + 1, pod_data_file, 8, 1.0)
    POD_path = score_builder._psi_pod
    edge_inPOD = score_builder.project_in_podspace(edge)
    score_visualizer = PODScoreVisualizer(40 + 1, 80 + 1, pod_data_file, 8, 1.0)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    list_tuples = []
    list_tuples.append([1, 2])
    list_tuples.append([1, 3])
    list_tuples.append([2, 3])
    list_tuples.append([1, 4])
    list_tuples.append([2, 4])
    list_tuples.append([3, 4])
    bounds = []
    bounds.append([0.45, 1.65])
    bounds.append([-1.55, 0.9])
    bounds.append([-0.35, 0.62])
    bounds.append([-0.35, 0.35])

    fsize = "xx-large"
    labels = ["No transition", "Transition"]

    plot_legend = True

    tidx = 0
    for dims in list_tuples:
        plt.figure(figsize=(6, 4.0))
        ctr = plt.contour(
            score_visualizer._map_coord_x[tidx, :],
            score_visualizer._map_coord_y[tidx, :],
            np.transpose(score_visualizer._maps[tidx, :, :]),
            levels=10,
            colors="grey",
            alpha=0.3,
        )
        plt.gca().clabel(ctr, fontsize=12)
        for i in (0, 1):
            mask = samples[:, 0].astype(int) == i
            plt.scatter(
                samples[mask, dims[0] + 1],
                samples[mask, dims[1] + 1],
                color=colors[samples[mask, 0].astype(int)],
                s=15,
                label=labels[i],
            )
        plt.plot(
            POD_path[:, dims[0] - 1],
            POD_path[:, dims[1] - 1],
            color=colors[2],
            linewidth=2.0,
            label=r"$\mathcal{P}_{tr}$",
        )
        plt.scatter(
            edge_inPOD[dims[0] - 1],
            edge_inPOD[dims[1] - 1],
            color="r",
            marker="*",
            s=150,
        )

        plt.grid(linestyle="dotted", color="silver", alpha=0.4)
        plt.xlabel(rf"$\mathbf{{v}}_{dims[0]}$", fontsize=fsize)
        plt.ylabel(rf"$\mathbf{{v}}_{dims[1]}$", fontsize=fsize)
        plt.xlim(left=bounds[dims[0] - 1][0], right=bounds[dims[0] - 1][1])
        plt.ylim(bottom=bounds[dims[1] - 1][0], top=bounds[dims[1] - 1][1])
        if plot_legend:
            plt.legend(fontsize=fsize)
            plot_legend = False
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.tight_layout()
        plt.show()
        plt.close()
        tidx += 1
