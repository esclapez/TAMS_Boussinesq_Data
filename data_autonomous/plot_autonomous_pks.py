"""Plot autononous forcing data."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    clist = list(
        colorsys.hls_to_rgb(c[0], min(1.0, max(0.0, 1 - amount * (1 - c[1]))), c[2])
    )
    clist.append(1.0)
    return np.array(clist)


if __name__ == "__main__":
    noise_levels = [0.0125, 0.01875, 0.025, 0.0375, 0.05]
    dataset = [
        "Naive/data_NOISE_NaiveNorth.npy",
        "Baars/data_NOISE_Baars.npy",
        "POD/data_NOISE_PODdata.npy",
    ]
    labels = [
        r"$\xi_1(\mathbf{X}_t)$",
        r"$\xi_2(\mathbf{X}_t)$",
        r"$\xi_3(\mathbf{X}_t)$",
    ]

    # Number of independent TAMS runs
    # and TAMS ensemble size
    nruns = 100
    N_TAMS = 25

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fsize = "x-large"

    colors = cm.viridis_r(np.linspace(0.0, 1, len(dataset)))
    colors[0] = lighten_color(colors[0], 1.2)

    plt.figure(figsize=(6, 4))
    for i in range(len(dataset)):
        data = np.zeros((len(noise_levels), 2))
        for n in range(len(noise_levels)):
            dsetfile = dataset[i].replace(
                "NOISE", str(noise_levels[n]).replace(".", "p")
            )
            if Path(dsetfile).exists():
                dset = np.load(dsetfile)
                data[n, 0] = dset[0, :].mean()
                data[n, 1] = dset[0, :].var()
            else:
                data[n, 0] = np.nan
                data[n, 1] = np.nan
        sigma_log = np.sqrt(np.log(1 + (np.sqrt(data[:, 1]) / (data[:, 0])) ** 2))
        CI_log = 1.96 * sigma_log / np.sqrt(nruns)
        plt.plot(
            noise_levels,
            data[:, 0],
            linewidth=1.5,
            color=colors[i],
            label=labels[i],
            linestyle="--",
            marker="o",
            markersize=4,
        )
        plt.fill_between(
            noise_levels,
            data[:, 0] * np.exp(CI_log),
            data[:, 0] / np.exp(CI_log),
            color=colors[i],
            alpha=0.2,
        )

    plt.grid(linestyle="dotted", color="silver")
    plt.xlim(left=0.01)
    plt.ylim(top=0.05, bottom=1.0e-10)
    plt.gcf().axes[0].set_yscale("log", base=10)
    plt.legend(fontsize=fsize)
    plt.xlabel(r"$\epsilon$", fontsize=fsize)
    plt.ylabel(r"$\overline{P}_K$", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    # plt.savefig("TransitionProba_darkerlog.pdf")
    plt.show()
    plt.close()

    # Lower and upper bounds on RE
    # MC: O(np.sqrt(1/(p N)))
    # TAMS: O(np.sqrt(log(1/p)/N))
    rel_error_bounds = np.zeros((9, 3))
    rel_error_bounds[:, 0] = np.logspace(-9, -1, 9)
    rel_error_bounds[:, 1] = np.sqrt(1 / (rel_error_bounds[:, 0] * N_TAMS))
    rel_error_bounds[:, 2] = np.sqrt(-np.log(rel_error_bounds[:, 0]) / N_TAMS)

    plt.figure(figsize=(6, 4))
    for i in range(len(dataset)):
        data = np.zeros((len(noise_levels), 2))
        for n in range(len(noise_levels)):
            dsetfile = dataset[i].replace(
                "NOISE", str(noise_levels[n]).replace(".", "p")
            )
            if Path(dsetfile).exists():
                dset = np.load(dsetfile)
                data[n, 0] = dset[0, :].mean()
                data[n, 1] = dset[0, :].var()
            else:
                data[n, 0] = np.nan
                data[n, 1] = np.nan
        rel_error = np.zeros(data.shape[0])
        rel_error[:] = np.sqrt(data[:, 1]) / data[:, 0]

        plt.plot(
            data[:, 0],
            rel_error[:],
            linewidth=1.0,
            color=colors[i],
            linestyle="--",
            marker="o",
            markersize=4,
            label=labels[i],
        )

    plt.plot(
        rel_error_bounds[:, 0],
        rel_error_bounds[:, 1],
        linewidth=1.0,
        label="Worst",
        color="black",
    )
    plt.plot(
        rel_error_bounds[:, 0],
        rel_error_bounds[:, 2],
        linewidth=1.0,
        label="Best",
        color="silver",
    )

    plt.gcf().axes[0].set_xscale("log", base=10)
    plt.gcf().axes[0].set_yscale("log", base=10)
    plt.grid(linestyle="dotted", color="silver")
    plt.legend(fontsize=fsize)
    plt.xlabel(r"$\overline{P}_K$", fontsize=fsize)
    plt.ylabel(r"RE$_K$", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.tight_layout()
    plt.show()
    # plt.savefig("RelErrors_darker.pdf")
    plt.close()
