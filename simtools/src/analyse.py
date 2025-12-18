import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_simulation_results(df, output_path="simtools/reports/figures/simulation_results.png"):
    """
    Create a plot with simulation results (mean and 95% CI).

    Parameters
    ----------
    df : pandas.DataFrame
        Results from run_experiment (columns: total_cards, mu, ci_lower, ci_upper).
    output_path : str
        Path where the plot is saved.
    """
    plt.figure(figsize=(10, 6))

    # Asymmetric error bars from confidence intervals
    yerr = np.vstack([
        df["mu"].to_numpy() - df["ci_lower"].to_numpy(),
        df["ci_upper"].to_numpy() - df["mu"].to_numpy(),
    ])

    plt.errorbar(
        df["total_cards"],
        df["mu"],
        yerr=yerr,
        fmt="o",
        label="Simulation (95% CI)",
        capsize=5,
    )

    plt.xlabel("Number of gold cards")
    plt.ylabel("Expected number of packs")
    plt.title("Simulation: packs needed to complete the collection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")


def plot_histogram(results, total_cards, output_path="simtools/reports/figures/histogram.png"):
    """
    Create a histogram of simulation results.

    Parameters
    ----------
    results : np.ndarray
        Array of pack counts per replication.
    total_cards : int
        Number of gold cards, used in the plot title.
    output_path : str
        Path where the plot is saved.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    mean_val = np.mean(results)
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                label=f"Mean: {mean_val:.1f}")
    plt.xlabel("Number of packs opened")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of packs needed ({total_cards} cards)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Histogram saved to: {output_path}")


def calculate_total_cost(expected_packs, pack_price):
    """
    Calculate the total cost to collect all cards.

    Parameters
    ----------
    expected_packs : float
        Expected number of packs (e.g. mean from simulation).
    pack_price : float
        Price per pack (e.g. 7500 coins or 1.25 €).

    Returns
    -------
    total_cost : float
        Total cost to collect all cards.
    """
    total_cost = expected_packs * pack_price
    return total_cost


if __name__ == "__main__":
    # Load simulation results
    df = pd.read_csv("simtools/reports/simulation_results.csv")

    # Create plot of simulation results
    plot_simulation_results(df)

    print("\nResult table:")
    print(df.to_string(index=False))

    # Select row for 1900 cards
    row_1900 = df.loc[df["total_cards"] == 1900].iloc[0]

    # Expected packs = mean over all replications for 1900 cards
    expected_packs = row_1900["mu"]

    # Costs in coins and points
    coins_cost = calculate_total_cost(expected_packs, pack_price=7500)   # one pack costs 7500 coins
    points_cost = calculate_total_cost(expected_packs, pack_price=1.25)  # one pack costs 1.25 €

    print(
        f"Total cost to collect all cards: "
        f"{coins_cost:.2f} coins or {points_cost:.2f} € in points"
    )
