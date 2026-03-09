"""Plot 2D maps of transient forcings data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    dataset = "POD"
    suffix = dataset

    # 2D maps parameters are alpha_0 and epsilon
    neps = 9
    nalpha = 7
    eps_range = np.linspace(0.001, 0.005, neps)
    alpha_range = np.linspace(0.45, 0.6, nalpha)

    # Number of run at each
    nruns = 100

    pmean_map = np.load(f"./{dataset}/pmean_map.npy")
    pRE_map = np.load(f"./{dataset}/pRE_map.npy")
    ext_map = np.load(f"./{dataset}/ext_map.npy")
    cost_map = np.load(f"./{dataset}/cost_map.npy")
    pRE_map_best = np.zeros(pmean_map.shape)
    for i in range(pmean_map.shape[0]):
        for j in range(pmean_map.shape[1]):
            if pmean_map[i, j] > 0.0:
                pRE_map_best[i, j] = np.sqrt(np.log(1 / pmean_map[i, j]) / nruns)
            else:
                pRE_map[i, j] = np.nan
                pRE_map_best[i, j] = np.nan

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    # Pmean map
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.imshow(
        np.flip(pmean_map, axis=0), cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=0.5)
    )
    plt.xlabel(r"$\epsilon$", fontsize="x-large")
    plt.ylabel(r"$\alpha_0$", fontsize="x-large")
    ax.set_xticks(np.arange(len(eps_range)))
    ax.set_xticklabels(np.round(eps_range, 4))
    ax.set_yticks(np.arange(len(alpha_range)))
    ax.set_yticklabels(np.flip(alpha_range))
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(pmean_map.shape[0]):
        for j in range(pmean_map.shape[1]):
            if np.flip(pmean_map, axis=0)[i, j] > 0.02:
                text = ax.text(
                    j,
                    i,
                    f"{np.flip(pmean_map, axis=0)[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="small",
                )
            else:
                if np.flip(pmean_map, axis=0)[i, j] > 0.001:
                    text = ax.text(
                        j,
                        i,
                        f"{np.flip(pmean_map, axis=0)[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize="small",
                    )
                else:
                    text = ax.text(
                        j,
                        i,
                        f"{np.flip(pmean_map, axis=0)[i, j]:.1e}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize="small",
                    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(label=r"$\overline{P}_{K=100}$", cax=cax)
    cbar.ax.tick_params(labelsize="x-large")
    cbar.ax.yaxis.label.set_size("x-large")

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"Pbar_map_{suffix}.pdf")
    plt.close()

    # pRE map
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.imshow(np.flip(pRE_map, axis=0), cmap="viridis", vmin=0, vmax=10)
    plt.xlabel(r"$\epsilon$", fontsize="x-large")
    plt.ylabel(r"$\alpha_0$", fontsize="x-large")
    ax.set_xticks(np.arange(len(eps_range)))
    ax.set_xticklabels(np.round(eps_range, 4))
    ax.set_yticks(np.arange(len(alpha_range)))
    ax.set_yticklabels(np.flip(alpha_range))
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(pmean_map.shape[0]):
        for j in range(pmean_map.shape[1]):
            if np.flip(pRE_map, axis=0)[i, j] > 3.0:
                text = ax.text(
                    j,
                    i,
                    f"{np.flip(pRE_map, axis=0)[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="small",
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{np.flip(pRE_map, axis=0)[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize="small",
                )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(label=r"$\mathrm{RE}_{100}$", cax=cax)
    cbar.ax.tick_params(labelsize="x-large")
    cbar.ax.yaxis.label.set_size("x-large")

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"PRE_map_{suffix}.pdf")
    plt.close()

    # ext map
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.imshow(np.flip(ext_map, axis=0), cmap="viridis", vmin=0, vmax=100)
    plt.xlabel(r"$\epsilon$", fontsize="x-large")
    plt.ylabel(r"$\alpha_0$", fontsize="x-large")
    ax.set_xticks(np.arange(len(eps_range)))
    ax.set_xticklabels(np.round(eps_range, 4))
    ax.set_yticks(np.arange(len(alpha_range)))
    ax.set_yticklabels(np.flip(alpha_range))
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(pmean_map.shape[0]):
        for j in range(pmean_map.shape[1]):
            if np.flip(ext_map, axis=0)[i, j] > 50.0:
                text = ax.text(
                    j,
                    i,
                    f"{np.flip(ext_map, axis=0)[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="small",
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{np.flip(ext_map, axis=0)[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize="small",
                )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(label=r"$N_{ext}$", cax=cax)
    cbar.ax.tick_params(labelsize="x-large")
    cbar.ax.yaxis.label.set_size("x-large")

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"Extinction_map_{suffix}.pdf")
    plt.close()
