"""
Validation helper functions.
"""

import pandas as pd


def is_missing(value):

    return pd.isna(value)