# HW5:
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Task 1: Define "titanic" dataset from seaborn lib.
import seaborn as sns
df = sns.load_dataset("titanic")

# Task 2: Find num. of female and male passengers from dataset
df["sex"].value_counts()

#Task 3: Find num. of unique values of each column
df.nunique()

# Task 4: Find num. of unique values of "pclass" var.
df["pclass"].nunique()

# Task 5: Find num. of unique values of "pclass" and "parch" vars.

df[["pclass", "parch"]].nunique() # çift [] olmalı

# Task 6: Check type of "embarked" var. Change its type to "category" and
# check again.
df["embarked"].dtype # returns dtype = "object"
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype # dtype = "category"

# Task 7: Show info of "embarked" var with value of "C"
df[df["embarked"] == "C"]

# Task 8: Show info of "embarked" var without value of "S"
df[df["embarked"] != "S"]

# Task 9: Show info of female passengers below the age of 30.
df[(df["sex"] == "female") & (df["age"] < 30)]

# Task 10: Show info of "fare" var bigger than 500 or passengers above the
# age of 70.

df[(df["fare"] > 500) | (df["age"] > 70)]

# Task 11: Find the sum of empty values in each var.
df.isnull().sum()

# Task 12: Drop "who" var. from dataframe
df.drop("who", axis=1, inplace=True)

# Task 13: Fill empty values of "deck" var with the most repeated value(mode)
# of "deck" var.
deck_most_rep = df["deck"].mode()[0] # or use value_counts instead of mode, same.
df["deck"].fillna(deck_most_rep, inplace=True)

# Task 14: Fill empty values of "age" var with the median of "age" var.
df["age"].fillna(df["age"].median(), inplace=True)

# Task 15: Find sum, count and mean values of "survived" var. with
# diffraction of "pclass" and "sex" vars.
diff_values = ["sum", "count", "mean"]
df.groupby(["pclass", "sex"]).agg({"survived": diff_values})

# Task 16: Write a func. that outputs 1 if age below 30, 0 else.
# Create a variable named "age_flag" in titanic dataset using this function.
# use apply and lambda

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# Task 17: Define "tips" dataset from "seaborn" lib.
import seaborn as sns
df = sns.load_dataset("tips")

# Task 18: Find sum, min, max and mean values of "total_bill" according
# to the "dinner" and "lunch" categories of "time" var.

df["time"].value_counts()
df.groupby(["time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# Task 19: Find sum, min, max and mean values of "total_bill" according
# to the "day" and "time"
df.head()
df.groupby(["time", "day"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# Task 20: Find sum, min, max and mean values of "total_bill" and "tip" in "day"
# according to the "lunch" time and "female" customers.

df[(df["time"] == "Lunch") & (df["sex"] == "Female")]\
    .groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                         "tip": ["sum", "min", "max", "mean"]}).iloc[0:2]

# Task 21: Find the mean of orders that has a size < 3 and total_bill > 10
df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].mean(numeric_only=True)

# Task 22: Create a new var. named "total_bill_tip_sum" which will
# give an output of every customer

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# Task 23:

mean_female = df[df["sex"] == "Female"]["total_bill"].mean()
mean_male = df[df["sex"] == "Male"]["total_bill"].mean()


def mean_sex(dataframe):
    if dataframe["sex"] == "Female":
        if dataframe["total_bill"] < mean_female:
            return 0
        else:
            return 1
    else:
        if dataframe["total_bill"] < mean_male:
            return 0
        else:
            return 1
df["total_bill_flag"] = df.apply(mean_sex, axis=1)

#### this part is not correct and will be edited later ####
df["total_bill_flag"] = df[df["sex"] == "Female"]["total_bill"].apply(lambda x: 0 if x < mean_female else 1)
df["total_bill_flag"] = df[df["sex"] == "Male"]["total_bill"].apply(lambda x: 0 if x < mean_male else 1)
#### this part is not correct and will be edited later ####





# Task 24: Using total_bill_flag var. observe whether values are above or below
# the average according to "sex"
df.groupby("sex")["total_bill_flag"].value_counts()

# Task 25: Take values in "total_bill_tip_sum" from descending sort and
# take first 30 values in different dataframe.

sorted_df = df.sort_values("total_bill_tip_sum", ascending=False)[0:30]