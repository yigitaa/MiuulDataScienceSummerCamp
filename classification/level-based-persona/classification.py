import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("...\persona.csv")
df.head()
df.info

# TASK 1 #
# Task 1.1: Read "persona.csv" and show general info about df.
df.info()
df.head()

# Task 1.2: How many unique SOURCE? What are frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Task 1.3: How many unique PRICE?
df["PRICE"].nunique()

# Task 1.4: How many sales happened each from PRICE.
df["PRICE"].value_counts()

# Task 1.5: How many sales done from COUNTRY.
df["COUNTRY"].value_counts()

# Task 1.6: Sum of sales with respect to COUNTRY.
df.groupby("COUNTRY")["PRICE"].sum()

# Task 1.7: Sale numbers according to SOURCE types.
df["SOURCE"].value_counts()

# Task 1.8: Avg. of PRICE acc. to COUNTRY types.
df.groupby("COUNTRY")["PRICE"].mean()

# Task 1.9: Avg. of PRICE acc. to SOURCE types.
df.groupby("SOURCE")["PRICE"].mean()

# Task 1.10: PRICE avg. in COUNTRY-SOURCE diffraction.
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# Task 2: Profit avg. in COUNTRY-SOURCE-SEX-AGE diffraction.
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

# Task 3: Sort the output from TASK 2 acc. to PRICE
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})\
    .sort_values("PRICE", ascending=False)

# Task 4: Convert index names to var. names
agg_df = agg_df.reset_index()

# Task 5: Change "AGE" var. to categorical var. and add to the agg_df.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40 ,70],
                           labels=["0_18", "19_23", "24_30", "31_40", "41_70"])

# Task 6: Define df persona

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_"
                                   + row[5].upper() + "_"  for row in agg_df.values]

agg_df["customers_level_based"].value_counts()

agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()
agg_df["customers_level_based"].value_counts()
# Task 7: Divide personas into segments.

#qcut: quantile cut
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

# Task 8:

new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user]["PRICE"].values)
print(agg_df[agg_df["customers_level_based"] == new_user]["SEGMENT"].values)



new_user2 = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user2]["PRICE"].values)
print(agg_df[agg_df["customers_level_based"] == new_user2]["SEGMENT"].values)

# segmentlere