import numpy as np
import pandas as pd
from time import perf_counter

# Wahrscheinlichkeit für die unterschiedlichen Ratings
def build_rating_probs(ratings):
    """
    ratings: 1D-Array mit Spieler-Ratings (ints 75–90).
    Returns: probs-Array gleicher Länge, Summe = 1.
    """
    ratings = np.asarray(ratings)

    # Klassenmasken
    low_rated_cards = (ratings <= 81)
    mid_rated_cards = (ratings >= 82) & (ratings <= 85)
    high_rated_cards = (ratings >= 86)

    # Ziel-Gesamtwahrscheinlichkeiten für die Klassen
    p_high_total = 0.05
    p_mid_total  = 0.232
    p_low_total  = 1.0 - p_high_total - p_mid_total  # Rest

    probs = np.zeros_like(ratings, dtype=float)

    # LOW: gleichverteilt in 75–81
    n_low = low_rated_cards.sum()
    if n_low > 0:
        probs[low_rated_cards] = p_low_total / n_low

    # MID: gleichverteilt in 82–85
    n_mid = mid_rated_cards.sum()
    if n_mid > 0:
        probs[mid_rated_cards] = p_mid_total / n_mid

    # HIGH (86+): hier wird innerhalb der Klasse noch nach Rating gewichtet
    # Idee: je höher das Rating, desto kleiner das Gewicht
    high_ratings = ratings[high_rated_cards]
    n_high = high_rated_cards.sum()
    if n_high > 0:
        # z.B. Gewicht ~ 1 / (rating - 80), damit 90er seltener als 84er
        raw_w = 1.0 / (high_ratings - 80)
        raw_w = raw_w / raw_w.sum()            # normieren innerhalb High
        probs[high_rated_cards] = p_high_total * raw_w

    # Numerische Sicherheit
    probs = probs / probs.sum()
    return probs

# Simulation des Pack Openings

def simulate_pack(total_cards, cards_per_pack, rng, probs):
    """
    Simuliert das Sammeln aller Karten in einem Durchlauf.
    
    Parameter
    ----------
    total_cards : int
        Gesamtanzahl unterschiedlicher Goldkarten.
    cards_per_pack : int
        Anzahl Goldkarten pro Pack.
    rng : np.random.Generator
        Zufallszahlengenerator.
    probs: array_like of float, shape (total_cards,)
        Ziehwahrscheinlichkeiten für jede Karte.
    
    Returns
    -------
    packs_opened : int
        Anzahl geöffneter Packs bis zur Komplettierung..

    """
    collected = set()
    packs_opened = 0
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()  # Sicherstellen, dass Summe 1 ist
    
    while len(collected) < total_cards:
        packs_opened += 1
        # Ziehe cards_per_pack zufällige Karten (mit Zurücklegen)
        new_cards = rng.choice(total_cards, size=cards_per_pack, replace=False, p=probs)
        collected.update(new_cards)
    
    return packs_opened


def mc_pack_opening(total_cards, cards_per_pack, n_replications, rsseq, probs):
    """
    Führt mehrere unabhängige Replikationen der Simulation durch.
    
    Parameter
    ----------
    total_cards : int
        Gesamtanzahl unterschiedlicher Goldkarten.
    cards_per_pack : int
        Anzahl Goldkarten pro Pack.
    n_replications : int
        Anzahl unabhängiger Replikationen.
    rsseq : np.random.SeedSequence
        Seed-Sequenz für reproduzierbare unabhängige Streams.
    probs: array_like of float, shape (total_cards,)
        Ziehwahrscheinlichkeiten für jede Karte.
    
    Returns
    -------
    results : np.ndarray
        Array mit Anzahl Packs pro Replikation.
    """
    # Erzeuge unabhängige Seeds für jede Replikation
    child_seeds = rsseq.spawn(n_replications)
    results = []
    
    for seed in child_seeds:
        rng = np.random.default_rng(seed)
        packs = simulate_pack(total_cards, cards_per_pack, rng, probs)
        results.append(packs)
    
    return np.array(results)


def run_experiment(card_counts, cards_per_pack, n_replications, rsseq):
    """
    Führt Experimente für verschiedene Kartenanzahlen durch.
    
    Parameter
    ----------
    card_counts : tuple of int
        Verschiedene Kartenanzahlen zu testen.
    cards_per_pack : int
        Anzahl Goldkarten pro Pack.
    n_replications : int
        Anzahl Replikationen pro Kartenanzahl.
    rsseq : np.random.SeedSequence
        Seed-Sequenz für Reproduzierbarkeit.
    
    Returns
    -------
    df : pd.DataFrame
        Ergebnisse mit Mittelwert, SE und Konfidenzintervallen.
    """
    rows = []
    
    # NOCHMAL ANSCHAUEN UND VERSTEHEN, WAS HIER PASSIERT
    for n_cards in card_counts:
        # Unabhängiger Seed-Stream für diese Kartenanzahl
        child_rsseq = rsseq.spawn(1)[0]
        rng = np.random.default_rng(child_rsseq)

        # Ratings für diese Kartenanzahl erzeugen
        ratings = rng.integers(75, 91, size=n_cards)
        probs = build_rating_probs(ratings)

        t0 = perf_counter()
        results = mc_pack_opening(
            total_cards=n_cards,
            cards_per_pack=cards_per_pack,
            n_replications=n_replications,
            rsseq=child_rsseq,
            probs=probs
        )
        dt = perf_counter() - t0
        
        # Statistiken berechnen
        mu = np.mean(results)
        se = np.std(results, ddof=1) / np.sqrt(n_replications)
        ki_untere_grenze = mu - 1.96 * se
        ki_obere_grenze = mu + 1.96 * se
        
        rows.append({
            'total_cards': n_cards,
            'cards_per_pack': cards_per_pack,
            'n_replications': n_replications,
            'mu': mu,
            'se': se,
            'ki_untere_grenze': ki_untere_grenze,
            'ki_obere_grenze': ki_obere_grenze,
            'time_s': dt
        })
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Experiment-Parameter
    CARD_COUNTS = (50, 100, 200, 500, 1000, 1900)
    CARDS_PER_PACK = 3
    N_REPLICATIONS = 100
    SEED = 42
    
    # Seed-Sequenz für Reproduzierbarkeit
    rsseq = np.random.SeedSequence(SEED)
    
    # Experiment durchführen
    results_df = run_experiment(CARD_COUNTS, CARDS_PER_PACK, N_REPLICATIONS, rsseq)
    
    # Ergebnisse speichern
    results_df.to_csv('simtools/reports/simulation_results.csv', index=False)
    print(results_df)