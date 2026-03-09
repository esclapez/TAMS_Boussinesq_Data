import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.interpolate import interp1d


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
    return colorsys.hls_to_rgb(c[0], min(1.0, max(0.0, 1 - amount * (1 - c[1]))), c[2])


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

    colorfactor = [1.5, 1.2, 1.0]
    colors = cm.viridis_r(np.linspace(0, 1, len(dataset)))
    avg_yspace = np.linspace(0.0, 1.0, 100)

    for s in range(len(dataset)):
        data = np.load(dataset[s])
        plt.figure(figsize=(4, 4))
        jtildes = np.zeros(data.shape[0])
        x_interp = []
        for i in range(data.shape[0]):
            mask = data[i, 0, :] > -1e8
            jtildes[i] = len(data[i, 0, mask])
            plt.plot(
                data[i, 0, mask],
                data[i, 1, mask],
                color=lighten_color(colors[s], np.random.uniform(0.4, 0.9, 1)[0]),
                alpha=0.8,
                linewidth=0.8,
            )
            plt.scatter(
                data[i, 0, mask],
                data[i, 2, mask],
                s=1,
                color=lighten_color(colors[s], colorfactor[s]),
            )
            f = interp1d(
                data[i, 1, mask],
                data[i, 0, mask],
                bounds_error=False,
                fill_value=np.nan,
            )
            x_interp.append(f(avg_yspace))

        x_interp = np.array(x_interp)
        plt.plot(
            np.nanmean(x_interp, axis=0),
            avg_yspace,
            color=lighten_color(colors[s], 1.1),
            linewidth=1.5,
            linestyle="--",
        )

        plt.grid(linestyle="dotted", color="silver")
        plt.xlim(left=0.0, right=400)
        plt.ylim(bottom=0.0, top=1.0)
        # plt.legend(fontsize=fsize)
        plt.xlabel(r"$J$", fontsize=fsize)
        plt.ylabel(r"$\mathcal{Q}_j$", fontsize=fsize)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        dist_jtilde = stats.gaussian_kde(jtildes, bw_method="silverman")
        jtilde_range = np.linspace(0, 1000, num=200)
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.fill_between(
            jtilde_range, dist_jtilde(jtilde_range), 0.0, color="silver", alpha=0.4
        )
        plt.ylim(bottom=0.0, top=0.1)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.ylabel(r"$\mathbf{P}_{J}$", fontsize=fsize)

        plt.tight_layout()
        # plt.savefig(f"MinMax_Span_{noise}_{dataset[s].split("/")[0]}.png", dpi=900)
        plt.show()
        plt.close()
