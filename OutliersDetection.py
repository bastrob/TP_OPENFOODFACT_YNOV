# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:39:44 2020

@author: Bast
"""
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutliersDetection(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        result = pd.DataFrame(X, columns=['energy_100g', 'fat_100g', 'carbohydrates_100g', 
                                          'sugars_100g', 'proteins_100g', 'salt_100g'])
        Q1 = result.quantile(q=.25)
        Q3 = result.quantile(q=.75)
        IQR = result.apply(stats.iqr)
        #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
        result = result[~((result < (Q1-1.5*IQR)) | (result > (Q3+1.5*IQR))).any(axis=1)]

        #find how many rows are left in the dataframe 
        return np.c_[result]