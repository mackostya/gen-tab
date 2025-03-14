import matplotlib.pyplot as plt


def hist_real_vs_fake_plot(real_data, fake_data, bins=100, name="Image") -> tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of real vs fake data.

    Args:
        real_data (np.ndarray): The real data.
        fake_data (np.ndarray): The fake data.
        bins (int): The number of bins.

    Returns:
        tuple: The figure and the axes.
    """
    fig, ax = plt.subplots()
    ax.hist(real_data, bins=bins, alpha=0.5, label="Real")
    ax.hist(fake_data, bins=bins, alpha=0.5, label="Fake")
    ax.set_title(f"{name}")
    ax.legend()
    return fig, ax
