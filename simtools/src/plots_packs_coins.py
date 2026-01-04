import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_packs_vs_cards(csv_path="reports/simulation_results.csv",
                                 output_path="reports/figures/packs_vs_cards.png"):
    """
    Plot expected packs vs. number of gold cards
    using Monte Carlo results from simulation.py.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)  # columns: total_cards, mu, ci_lower, ci_upper, ...

    x = df["total_cards"].to_numpy()
    y = df["mu"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linewidth=2, markersize=8, label="Simulation mean")

    # highlight 1900 if present
    if (df["total_cards"] == 1900).any():
        mu_1900 = df.loc[df["total_cards"] == 1900, "mu"].iloc[0]
        plt.scatter([1900], [mu_1900],
                    color="red", s=100, zorder=5, label="FIFA 26: 1900 cards")

    plt.xlabel("Number of gold cards", fontsize=12)
    plt.ylabel("Expected number of packs (simulation)", fontsize=12)
    plt.title("Simulated packs needed to collect all gold cards", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")
    

def plot_coins_vs_cards(
    csv_path="reports/simulation_results.csv",
    output_path="reports/figures/coins_vs_cards.png",
    pack_price=7500,
):
    """
    Plot expected coins (in millions) vs. number of gold cards
    using Monte Carlo results from simulation.py.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    x = df["total_cards"].to_numpy()
    packs = df["mu"].to_numpy()
    coins_mio = packs * pack_price / 1_000_000

    plt.figure(figsize=(10, 6))
    plt.plot(x, coins_mio, "o-", linewidth=2, markersize=8, color="orange",
             label="Simulation mean")

    if (df["total_cards"] == 1900).any():
        mu_1900 = df.loc[df["total_cards"] == 1900, "mu"].iloc[0]
        coins_1900_mio = mu_1900 * pack_price / 1_000_000
        plt.scatter([1900], [coins_1900_mio],
                    color="red", s=100, zorder=5, label="FIFA 26: 1900 cards")

    plt.xlabel("Number of gold cards", fontsize=12)
    plt.ylabel("Expected cost (million coins)", fontsize=12)
    plt.title(
        f"Simulated coins needed to collect all gold cards\n({pack_price:,.0f} coins per pack)",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")



if __name__ == "__main__":
    plot_packs_vs_cards()
    plot_coins_vs_cards()
    
    # Calculate costs for 1900 cards
    expected_packs = pd.read_csv("reports/simulation_results.csv").loc[
        lambda df: df["total_cards"] == 1900, "mu"
    ].iloc[0]
    coins_cost = expected_packs * 7500
    points_cost = expected_packs * 1.25
    
    print(f"\nExpected packs for 1900 cards: {expected_packs:.2f}")
    print(f"Total cost: {coins_cost:.2f} coins or {points_cost:.2f} € in points")
