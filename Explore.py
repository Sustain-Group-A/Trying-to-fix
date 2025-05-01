from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

#read in processed csv to a dataframe
df = pd.read_csv("full_grain_data.csv")
df['grain_type'] = df['grain_type'].str.extract(r"([^/]+)(?=-\d+\.hdr$)", expand=False)
print(df)

#check for number of rice classes
print("Number of classes:")
print(len(df['grain_type'].unique()))

print(df.info())
print(df.describe())
dfnull = df.isnull().sum()
print("Checking for missing data")
print(dfnull)
for i in dfnull:
    if dfnull[i] > 0:
        print("Missing data! row:", i)
    else:
        pass
       # print("Test line")#if there is no missing data nothing will print, this line was used to test the function works.

df1 = (df.groupby(['grain_type'])['area'].agg(lambda x: x.unique().mean()))
df2 = (df.groupby(['grain_type'])['eccentricity'].agg(lambda x: x.unique().mean()))
df3 = (df.groupby(['grain_type'])['perimeter'].agg(lambda x: x.unique().mean()))

fig1 = plt.figure()
plt.title("Average size for each spieces")
plt.xlabel("Spieces Name")
plt.ylabel("Size")
plt.plot(df1)
plt.show()

fig2 = plt.figure()
plt.title("Average Eccentricity for each spieces")
plt.xlabel("Spieces Name")
plt.ylabel("Eccentricity")
plt.plot(df2)
plt.show()

fig3 = plt.figure()
plt.title("Average Perimeter for each spieces")
plt.xlabel("Spieces Name")
plt.ylabel("Perimeter")
plt.plot(df3)
plt.show()

fig4 = plt.figure()
plt.title("Histogram for grain types")
plt.xlabel("Grain Type")
plt.ylabel("Frequency")
plt.hist(df['grain_type'])
plt.show()