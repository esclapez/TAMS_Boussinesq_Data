"""Plot number of transitions from TAMS and DNS."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    suffix = ""
    dataset = ["Naive", "Baars", "POD"]
    labels = [
        r"$\xi_1(\mathbf{X}_t)$",
        r"$\xi_2(\mathbf{X}_t)$",
        r"$\xi_3(\mathbf{X}_t)$",
    ]

    # The baseline cost of a DNS trajectory: 4000 time steps
    base_cost_traj = 4.0e3

    # Number of run at each
    N_TAMS = 25
    nruns = 100

    colors = cm.viridis_r(np.linspace(0, 1, len(dataset)))
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fsize = "x-large"
    plt.figure(figsize=(6, 4))
    plt.grid(linestyle="dotted", color="silver", alpha=0.8)

    for i in range(len(dataset)):
        pmean_map = np.load(f"./{dataset[i]}/pmean_map{suffix}.npy")
        pRE_map = np.load(f"./{dataset[i]}/pRE_map{suffix}.npy")
        ext_map = np.load(f"./{dataset[i]}/ext_map{suffix}.npy")
        Ntrans_TAMS = (nruns - ext_map) * N_TAMS
        MC_equil = (
            np.sqrt((1 - pmean_map) / (pmean_map * pRE_map**2) * base_cost_traj)
            * pRE_map
        )
        Ntrasn_MC = (1 - pmean_map) / (pmean_map * pRE_map**2) * pmean_map
        plt.scatter(pmean_map, Ntrans_TAMS, color=colors[i], label=labels[i], s=15)
        plt.scatter(pmean_map, Ntrasn_MC, color="r", s=10)

    plt.xlim(left=1e-8, right=1.0)
    plt.ylim(bottom=5e-3, top=5e3)
    plt.gcf().axes[0].set_xscale("log", base=10)
    plt.gcf().axes[0].set_yscale("log", base=10)
    plt.hlines(1.0, 1e-8, 1.0, color="k", linestyle="--")

    plt.xlabel(r"$\overline{P}_K$", fontsize=fsize)
    plt.ylabel(r"$N_{transit}$", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.show()
    # plt.savefig("Ntransit_withDNS.pdf")
