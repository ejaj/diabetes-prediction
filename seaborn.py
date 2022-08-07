import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

with open('data/data.txt') as csvfile:
    population, profit = zip(*[(float(row['Population']), float(row['Profit'])) for row in csv.DictReader(csvfile)])

df = pd.DataFrame()
df['Population'] = population
df['Profit'] = profit
sns.lmplot(x="Population", y="Profit", data=df, fit_reg=False, scatter_kws={'s': 45})
plt.show()
