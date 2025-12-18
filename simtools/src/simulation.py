import numpy as np
import pandas as pd
from time import perf_counter


# Probability for different ratings
def build_rating_probs(ratings):
    """
    Build draw probabilities for players based on their ratings.

    Parameters
    ----------
    ratings : array_like of int, shape (n_cards,)
        Player ratings (e.g. integers 75–90).

    Returns
    -------
    probs : ndarray of float, shape (n_cards,)
        Draw probabilities for each card, summing to 1.
    """
    ratings = np.asarray(ratings)

    # Rating class masks
    low_rated_cards = (ratings <= 81)
    mid_rated_cards = (ratings >= 82) & (ratings <= 85)
    high_rated_cards = (ratings >= 86)

    # Target total probabilities for each class
    p_high_total = 0.05
    p_mid_total = 0.232
    p_low_total = 1.0 - p_high_total - p_mid_total  # remaining probability

    probs = np.zeros_like(ratings, dtype=float)

    # LOW: uniformly distributed within 75–81
    n_low = low_rated_cards.sum()
    if n_low > 0:
        probs[low_rated_cards] = p_low_total / n_low

    # MID: uniformly distributed within 82–85
    n_mid = mid_rated_cards.sum()
    if n_mid > 0:
        probs[mid_rated_cards] = p_mid_total / n_mid

    # HIGH (86+): within this class we weight by rating
    # Idea: the higher the rating, the smaller the weight
    high_ratings = ratings[high_rated_cards]
    n_high = high_rated_cards.sum()
    if n_high > 0:
        # Example: weight ~ 1 / (rating - 80),
        # so 90s are rarer than 86s (1 / 10 vs. 1 / 6)
        raw_w = 1.0 / (high_ratings - 80)
        raw_w = raw_w / raw_w.sum()           # normalize within high class
        probs[high_rated_cards] = p_high_total * raw_w

    # Numerical safety: ensure probabilities sum to 1
    probs = probs / probs.sum()
    return probs


# Simulation of pack opening
def simulate_pack(total_cards, cards_per_pack, rng, probs):
    """
    Simulate collecting all cards in a single run.

    Parameters
    ----------
    total_cards : int
        Total number of distinct gold cards.
    cards_per_pack : int
        Number of gold cards per pack.
    rng : np.random.Generator
        Random number generator.
    probs : array_like of float, shape (total_cards,)
        Draw probabilities for each card.

    Returns
    -------
    packs_opened : int
        Number of packs opened until the collection is complete.
    """
    collected = set()
    packs_opened = 0
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()  # ensure probabilities sum to 1

    while len(collected) < total_cards:
        packs_opened += 1
        # Draw cards_per_pack cards (without replacement within a pack)
        new_cards = rng.choice(
            total_cards,
            size=cards_per_pack,
            replace=False,
            p=probs
        )
        collected.update(new_cards)

    return packs_opened


def mc_pack_opening(total_cards, cards_per_pack, n_replications, rsseq, probs):
    """
    Run multiple independent replications of the pack-opening simulation.

    Parameters
    ----------
    total_cards : int
        Total number of distinct gold cards.
    cards_per_pack : int
        Number of gold cards per pack.
    n_replications : int
        Number of independent replications.
    rsseq : np.random.SeedSequence
        Seed sequence for reproducible independent random streams.
    probs : array_like of float, shape (total_cards,)
        Draw probabilities for each card.

    Returns
    -------
    results : ndarray of int, shape (n_replications,)
        Number of packs needed in each replication.
    """
    # Create independent seeds for each replication
    child_seeds = rsseq.spawn(n_replications)
    results = []

    for seed in child_seeds:
        rng = np.random.default_rng(seed)
        packs = simulate_pack(total_cards, cards_per_pack, rng, probs)
        results.append(packs)

    return np.array(results)


def run_experiment(card_counts, cards_per_pack, n_replications, rsseq):
    """
    Run experiments for several different numbers of total cards.

    Parameters
    ----------
    card_counts : tuple of int
        Different total card counts to test.
    cards_per_pack : int
        Number of gold cards per pack.
    n_replications : int
        Number of replications per card count.
    rsseq : np.random.SeedSequence
        Seed sequence for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Results with mean, standard error, and confidence intervals.
    """
    rows = []

    for n_cards in card_counts:
        # Independent seed stream for this card count
        child_rsseq = rsseq.spawn(1)[0]
        rng = np.random.default_rng(child_rsseq)

        # Generate ratings for this card count
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

        # Compute statistics
        mu = np.mean(results)
        se = np.std(results, ddof=1) / np.sqrt(n_replications)
        ci_lower = mu - 1.96 * se
        ci_upper = mu + 1.96 * se

        rows.append({
            "total_cards": n_cards,
            "cards_per_pack": cards_per_pack,
            "n_replications": n_replications,
            "mu": mu,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "time_s": dt,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Experiment parameters
    CARD_COUNTS = (50, 100, 200, 500, 1000, 1900)
    CARDS_PER_PACK = 3
    N_REPLICATIONS = 100
    SEED = 42

    # Seed sequence for reproducibility
    rsseq = np.random.SeedSequence(SEED)

    # Run experiment
    results_df = run_experiment(
        CARD_COUNTS,
        CARDS_PER_PACK,
        N_REPLICATIONS,
        rsseq
    )

    # Save results
    results_df.to_csv("simtools/reports/simulation_results.csv", index=False)
    print(results_df)
