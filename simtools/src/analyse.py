import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOCHMAL ANSCHAUEN UND VERSTEHEN, WAS HIER PASSIERT
# expected_packs wird nicht explizit ausgegeben
def coupon_collector_expected(total_cards, cards_per_pack):
    """
    Analytische Berechnung der erwarteten Packs (Coupon Collector).
    
    Parameters
    ----------
    total_cards : int
        Gesamtanzahl unterschiedlicher Goldkarten.
    cards_per_pack : int
        Anzahl Goldkarten pro Pack.
    
    Returns
    -------
    expected_packs : float
        Erwartete Anzahl Packs.
    """
    expected_draws = total_cards * np.log(total_cards)
    expected_packs = expected_draws / cards_per_pack
    return expected_packs

# Plotten der Ergebnisse
def plot_simulation_results(df, output_path='simtools/reports/figures/simulation_vs_theory.png'):
    """
    Erstellt Plot mit Simulationsergebnissen und theoretischen Werten.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ergebnisse aus run_experiment.
    output_path : str
        Pfad zum Speichern des Plots.
    """
    # Theoretische Werte berechnen (mit Formel von oben)
    df['theory'] = df.apply(
        lambda row: coupon_collector_expected(row['total_cards'], row['cards_per_pack']),
        axis=1
    )
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Simulation mit Fehlerbalken (yerr = y-error)
    yerr = np.vstack([
    df['mu'].to_numpy() - df['ki_untere_grenze'].to_numpy(),
    df['ki_obere_grenze'].to_numpy() - df['mu'].to_numpy()
])

    plt.errorbar(
        df['total_cards'],
        df['mu'],
        yerr=yerr,
        fmt='o',
        label='Simulation (95% CI)',
        capsize=5
    )
    
    # Theoretische Kurve
    plt.plot(df['total_cards'], df['theory'], 'r--', label='Theorie (Coupon Collector)')
    
    plt.xlabel('Anzahl Goldkarten')
    plt.ylabel('Erwartete Anzahl Packs')
    plt.title('Simulation vs. Theorie: Packs bis zur Komplettierung')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot gespeichert unter: {output_path}")


def plot_histogram(results, total_cards, output_path='simtools/reports/figures/histogram.png'):
    """
    Erstellt Histogramm der Simulationsergebnisse.
    
    Parameters
    ----------
    results : np.ndarray
        Array mit Anzahl Packs pro Replikation.
    total_cards : int
        Anzahl der Goldkarten für Titel.
    output_path : str
        Pfad zum Speichern des Plots.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(results), color='red', linestyle='--', linewidth=2, label=f'Mittelwert: {np.mean(results):.1f}')
    plt.xlabel('Anzahl geöffneter Packs')
    plt.ylabel('Häufigkeit')
    plt.title(f'Verteilung der benötigten Packs ({total_cards} Karten)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogramm gespeichert unter: {output_path}")


if __name__ == '__main__':
    # Ergebnisse laden
    df = pd.read_csv('simtools/reports/simulation_results.csv')
    
    # Plot erstellen
    plot_simulation_results(df)
    
    print("\nErgebnistabelle:")
    print(df.to_string(index=False))

# Kosten für alle Karten

# Ergebnisse aus der Simulation laden
df = pd.read_csv("simtools/reports/simulation_results.csv")

# Zeile für 1900 Karten auswählen
row_1900 = df.loc[df["total_cards"] == 1900].iloc[0]

expected_packs = row_1900["mu"]   # Mittelwert über alle Replikationen

def calculate_total_cost(expected_packs, pack_price):
    """
    Berechnet die Gesamtkosten für das Sammeln aller Karten.
    
    Parameters
    ----------
    total_cards : int
        Gesamtanzahl unterschiedlicher Goldkarten.
    cards_per_pack : int
        Anzahl Goldkarten pro Pack.
    pack_price : float
        Preis pro Pack.
    expected_packs : float
        Erwartete Anzahl Packs.
    
    Returns
    -------
    total_cost : float
        Gesamtkosten für das Sammeln aller Karten.
    """
    total_cost = expected_packs * pack_price
    return total_cost

coins_kosten = calculate_total_cost (expected_packs, pack_price=7500)
points_kosten = calculate_total_cost (expected_packs, pack_price=1.25)

print(f'Gesamtkosten für das Sammeln aller Karten: {coins_kosten:.2f} Coins oder {points_kosten:.2f} € für Points')