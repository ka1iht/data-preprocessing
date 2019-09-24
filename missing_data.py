# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
m_values = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
m_values = m_values.fit(X[:, 1:3])
X[:, 1:3] = m_values.transform(X[:, 1:3])