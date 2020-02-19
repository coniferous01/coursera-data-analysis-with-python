import pandas as pd
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = ['symboling', 'normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

df = pd.read_csv(url, header = None)
df.columns = headers

# ### Create csv file.
# path = "c:/Users/John J/Documents/Programming/Coursera_DataAnPy_CarData.csv"
#df.to_csv(path)

# ### Inspect data.
# print(df.describe(include="all"))
# print(df.head())

### Replace "?" characters with NaN.
df = df.replace("?", np.nan)

# ### Create data frame to assess missing values.
# missing_data = df.isnull()
# print(missing_data.head())
#
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())

### Replace missing values with mean.
mean_normloss = df['normalized-losses'].astype('float').mean(axis=0)
mean_bore = df['bore'].astype('float').mean(axis = 0)
mean_stroke = df['stroke'].astype('float').mean(axis = 0)
mean_horse = df['horsepower'].astype('float').mean(axis = 0)
mean_peak = df['peak-rpm'].astype('float').mean(axis = 0)

df['normalized-losses'].replace(np.nan, mean_normloss, inplace = True)
df['bore'].replace(np.nan, mean_bore, inplace = True)
df['stroke'].replace(np.nan, mean_stroke, inplace = True)
df['horsepower'].replace(np.nan, mean_horse, inplace = True)
df['peak-rpm'].replace(np.nan, mean_peak, inplace = True)

### Replace missing values with most frequent value.
df['num-of-doors'].replace(np.nan, "four", inplace=True)

### Drop rows where price value is missing.
df.dropna(subset = ['price'], axis = 0, inplace = True)

df.reset_index(drop=True, inplace=True)

# for column in df:
    # print(column)
    # print(df[column].describe(include="all"))

# missing_data = df.isnull()
# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())

# print(df.dtypes)

### Change data type from 'object' to 'float' for specified columns.
df[["normalized-losses",'stroke', 'bore', "horsepower", 'peak-rpm', 'price']]= df[["normalized-losses",'stroke', 'bore', "horsepower", 'peak-rpm', 'price']].astype("float")

# print(df[["normalized-losses", "horsepower", 'peak-rpm', 'price']].describe(include = 'all'))

### Define functions for normalizing columns.
def simple_feature_scaling(column):
    df[column] = df[column]/df[column].max()

def min_max(column):
    df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())

def z_score(column):
    df[column] = (df[column]-df[column].mean())/df[column].std()

# z_score('city-mpg')

# print(df.head())