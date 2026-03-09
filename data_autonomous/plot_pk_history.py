"""Plot P_K history from autonomous forcing cases."""

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
    # Noise level to plot: one of [0.05, 0.025, 0.0125]
    noise = "0p0125"
    dataset = [
        f"Naive/stats_{noise}_NaiveNorth.npy",
        f"Baars/stats_{noise}_Baars.npy",
        f"POD/stats_{noise}_PODdata.npy",
    ]
    labels = [
        r"$\xi_1(\mathbf{X}_t)$",
        r"$\xi_2(\mathbf{X}_t)$",
        r"$\xi_3(\mathbf{X}_t)$",
    ]
    nrun = 100

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fsize = "x-large"

    colors = cm.viridis_r(np.linspace(0, 1, len(dataset)))
    colors[0] = lighten_color(colors[0], 1.2)

    plt.figure(figsize=(4.4, 4))
    for i in range(len(dataset)):
        data = np.load(dataset[i])
        K_array = np.linspace(1, data.shape[1], data.shape[1])
        sigma = data[1, :] * np.sqrt(K_array)
        sigma_log = np.sqrt(np.log(1 + (sigma[:] / data[0, :]) ** 2))
        CI_log = 1.96 * sigma_log / np.sqrt(K_array)
        plt.plot(
            np.linspace(1, nrun, nrun),
            data[0, :],
            linewidth=1.5,
            color=colors[i],
            label=labels[i],
            linestyle="--",
        )
        plt.fill_between(
            np.linspace(1, nrun, nrun),
            data[0, :] * np.exp(CI_log),
            data[0, :] / np.exp(CI_log),
            color=colors[i],
            alpha=0.2,
        )

    plt.grid(linestyle="dotted", color="silver")
    plt.xlim(left=0.0)
    plt.legend(fontsize=fsize)
    plt.xlabel(r"$K$", fontsize=fsize)
    plt.ylabel(r"$\overline{P}_K$", fontsize=fsize)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    # plt.savefig(f"P_K_history_{noise}.pdf")
    plt.show()
