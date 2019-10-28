import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print('----prints everything----')
print(cancer)

print('----only the data ----')
print(cancer.data)

print('----shape of the data: 569, 30----')
print(cancer.data.shape)
cancer_data = np.c_[cancer.data, cancer.target]

print('----target names: malignant, benign----')
print(cancer.target_names)

print('----we have 30 for col_name----')
col_name = cancer.feature_names
print(col_name)

print('----new col names, now have 31----')
col_name = np.append(col_name, 'last-col')
print(col_name)

print('----now we have the df_cancer----')
df_cancer = pd.DataFrame(cancer_data, columns=col_name)
print(df_cancer)

print('-------Reach-------')
print(df_cancer['last-col'])
os.makedirs('plots', exist_ok=True)

plt.plot(df_cancer['mean area'], color='red')
plt.title('mean area')
plt.xlabel('Index')
plt.ylabel('last col')
plt.savefig(f'plots/mean_area_by_index_plot.png', format='png')
plt.clf()

# Plotting histogram
plt.hist(df_cancer['worst area'], bins=3, color='g')
plt.title('Worst Area')
plt.xlabel('Worst Area')
plt.ylabel('Count')
plt.savefig(f'plots/worst-area.png', format='png')
plt.clf()


# Plotting scatterplot - mean area and worst area
plt.scatter(df_cancer['mean area'], df_cancer['worst area'], color='b')
plt.title('Mean Area vs worst Area')
plt.xlabel('Mean Area')
plt.ylabel('worst Area')
plt.savefig(f'plots/mean_worst.png', format='png')

#using arg
arg1 = sys.argv[0]   # name of script
arg2 = sys.argv[1]       # one of the column
print(arg2)
plt.plot(df_cancer[arg2], color='red')
plt.title(arg2)
plt.xlabel('Index')
plt.ylabel(arg2)
plt.savefig(f'plots/' + arg2 + '.png', format='png')
plt.clf()