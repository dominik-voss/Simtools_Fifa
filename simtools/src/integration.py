import numpy as np

def mc_expectation(g, sampler, N, rng):
    """
    Schätzt E[g(X)] = Integral of g(x) f(x) dx via Monte Carlo.
    
    Returns
    -------
    mu : float
        MC-Schätzer (Mittelwert der g(X_i)).
    se : float
        Standardfehler (sd/sqrt(n)).
    """
    x = sampler(N, rng)
    gx = g(x)
    mu = gx.mean()
    se = gx.std(ddof=1) / np.sqrt(N)
    return mu, se
