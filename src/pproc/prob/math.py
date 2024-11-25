import numpy as np
import numexpr
from typing import Dict


def ensemble_probability(data: np.array, threshold: Dict) -> np.array:
    """Ensemble Probabilities:

    Computes the probability of a given parameter crossing a given threshold,
    by checking how many times it occurs across all ensembles.
    e.g. the chance of temperature being less than 0C

    """

    # Find all locations where np.nan appears as an ensemble value
    is_nan = np.isnan(arr).any(axis=0)

    # Read threshold configuration and compute probability
    comparison = threshold["comparison"]
    comp = numexpr.evaluate(
        "data " + comparison + str(threshold["value"]), local_dict={"data": data}
    )
    probability = np.where(comp, 100, 0).mean(axis=0)

    # Put in missing values
    probability = np.where(is_nan, np.nan, probability)

    return probability
