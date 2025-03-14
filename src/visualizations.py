import matplotlib.pyplot as plt
import numpy as np


def hist_real_vs_fake_plot(real_data, fake_data, bins=100, name="Image") -> tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of real vs fake data with enhanced aesthetics.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure there are no NaNs in real_data
    real_data = np.where(np.isnan(real_data), 0, real_data)

    # Calculate plot limits based on real_data range
    band = np.max(real_data) - np.min(real_data)
    xlim_min = np.min(real_data) - band / 4
    xlim_max = np.max(real_data) + band / 4

    # Plot the histograms with density normalization, custom colors, and edge styling
    ax.hist(
        real_data, bins=bins, density=True, alpha=0.6, label="Real", edgecolor="black", linewidth=1.0, color="#4c72b0"
    )
    ax.hist(
        fake_data, bins=bins, density=True, alpha=0.6, label="Fake", edgecolor="black", linewidth=1.0, color="#dd8452"
    )

    # Set title and axis labels with enhanced fonts
    ax.set_title(name, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Value", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)

    # Set x-axis limits and scientific notation for y-axis
    ax.set_xlim([xlim_min, xlim_max])
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add grid lines manually
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Customize legend
    legend = ax.legend(frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor("gray")

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust tick parameters for better readability
    ax.tick_params(axis="both", which="major", labelsize=12)

    return fig, ax
