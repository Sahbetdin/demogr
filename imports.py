import pandas as pd
pd.options.display.max_columns=500
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns

# import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from sklearn import ensemble, datasets, tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import xgboost as xgb

#miport each column meaning
from feat_meaning import meanings as d