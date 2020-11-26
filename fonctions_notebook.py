# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:51:43 2020

@author: Bast
"""

import pandas as pd


def load_csv(path: str, sep=',') -> pd.DataFrame:
    return pd.read_csv(path, sep)