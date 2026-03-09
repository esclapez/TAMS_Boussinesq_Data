"""Plot the committor-like behavior of the scores."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def extract_XY(
    lk_runs: list[np.ndarray],
    zk_runs: list[np.ndarray],
    N: int,
):
    """
    Extract X = (z_{k-1}/z_k) and
            Y = (1 - l_k/N)
    grouped by iteration index k.

    Returns
    -------
    X_by_k, Y_by_k : lists of lists
        X_by_k[k] = [X_k^{(d)} for all runs d having iteration k]
    """
    D = len(lk_runs)
    K_max = max(len(lk) for lk in lk_runs)
    # K_min = min(len(lk) for lk in lk_runs)

    X_by_k = [[] for _ in range(K_max)]
    Y_by_k = [[] for _ in range(K_max)]

    for d in range(D):
        lk = lk_runs[d]
        z = zk_runs[d]

        for k in range(len(lk)):
            p_hat = 1.0 - lk[k] / N
            if p_hat <= 0:
                continue  # safeguard

            r = z[k] / z[k + 1]
            if r <= 0:
                continue

            X_by_k[k].append(r)
            Y_by_k[k].append(p_hat)

    return X_by_k, Y_by_k


if __name__ == "__main__":
    # Noise level to plot: one of [0.05, 0.025, 0.0125]
    noise = "0p05"
    dataset = [
        f"Naive/minmax_{noise}_NaiveNorth.npy",
        f"Baars/minmax_{noise}_Baars.npy",
        f"POD/minmax_{noise}_PODdata.npy",
    ]
    labels = [
        r"$\xi_1(\mathbf{X}_t)$",
        r"$\xi_2(\mathbf{X}_t)$",
        r"$\xi_3(\mathbf{X}_t)$",
    ]
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fsize = "x-large"
    colors = cm.viridis_r(np.linspace(0, 1, len(dataset)))

    # TAMS ensemble size
    N_TAMS = 25

    plt.figure(figsize=(5.0, 4.0))
    for s in range(len(dataset)):
        data = np.load(dataset[s])
        errors = []
        betas = []
        res_vars = []
        lks_data = []
        zks_data = []
        interpolated_drifts = []
        for i in range(data.shape[0]):
            masked_data = data[i, :, data[i, 0, :] > -1e8]
            lks_data.append(np.diff(masked_data[:, 0]))
            zks_data.append(masked_data[:, 1])

        R_by_k, P_by_k = extract_XY(lks_data, zks_data, N_TAMS)
        meanR_by_k = np.array([np.array(arr).mean() for arr in R_by_k])
        meanP_by_k = np.array([np.array(arr).mean() for arr in P_by_k])
        mismatch = np.abs(meanR_by_k - meanP_by_k)
        d_mismatch = np.abs(np.diff(mismatch))
        plt.plot(
            np.linspace(1, len(R_by_k), len(R_by_k) - 1),
            np.cumsum(d_mismatch),
            color=colors[s],
            label=labels[s],
        )

    plt.ylim(top=12.0)
    plt.grid(linestyle="--", color="silver", alpha=0.5)
    plt.ylabel(r"$\sum_{i=2}^j |d_{i} - d_{i-1}|$", fontsize=fsize)
    plt.xlabel(r"$j$", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"CummulativeSum_derdiff_{noise}.pdf")
