# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:43:00 2018

@author: deepak
"""

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dir = "D:\\filepath\\filename.csv"

d = os.path.abspath(dir)

df = pd.read_csv(dir, header=0, encoding='latin1', low_memory=False)

# Subset the dataset into all the numerical values       
numeric_df = df.select_dtypes(include=[np.number])

# Complete the correlation matrix
corr = numeric_df._get_numeric_data().corr()

# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than <value>
to_drop = [column for column in upper.columns if any(upper[column] > 0.45)]

print(len(to_drop))

df1 = df.drop(to_drop, axis=1)

numeric_df1 = df1.select_dtypes(include=[np.number])

# Complete the correlation matrix
corr1 = numeric_df1._get_numeric_data().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr1, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr1, mask=mask, cmap=cmap, center=0.0,
                      vmax = 1, square=True, linewidths=.5, ax=ax)
plt.savefig('corr-heat.png')
plt.show()

