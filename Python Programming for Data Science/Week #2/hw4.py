# HW4:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

# Q1: Add function parameter

# Task 1:  Add one parameter to cat_summary() function and make sure
# it is modifiable.

# Original cat_summary function
def cat_summary(dataframe, col_name):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")

cat_summary(df,"sex")

# Edited
def cat_summary(dataframe, col_name, plot=False):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                  "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

# Q2: Docstring

# Task 2: Write four(if suitable) definitions(task, params, return, example) to check_df() and cat_summary() functions
def check_df(dataframe, head=5):
    """

    Parameters:
    ----------
    dataframe: dataframe [str]
    head: number of data per column, [int]

    Returns
    -------

    Prints df shape, dtype, head, tail, isnull and quantile
    """
    print("################ Shape #################")
    print(dataframe.shape)
    print("################ Types #################")
    print(dataframe.dtypes)
    print("################ Head ##################")
    print(dataframe.head(head))
    print("################ Tail ##################")
    print(dataframe.tail(head))
    print("################ NA ####################")
    print(dataframe.isnull().sum())
    print("################ Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
    print("#################### END ###################")

check_df(df, head=5)

def cat_summary(dataframe, col_name, plot=False):
    """

    Parameters
    ----------
    dataframe: dataframe [str]
    col_name: column name [str]
    plot: boolean

    Returns
    -------
    Value of col_name and its ratio of classes from given dataframe

    Example
    -------
    ######################################
            sex      Ratio
    male    577  64.758698
    female  314  35.241302
    ######################################
    """

    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                  "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)