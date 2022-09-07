import numpy as np
import pandas as pd
import datetime as dt
from warnings import filterwarnings

filterwarnings('ignore')
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_excel(".../online_retail_II.xlsx")


################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def retail_data_prep(dataframe):
    return dataframe


df = retail_data_prep(df)
check_df(df)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

# Since I cannot get in touch with the company, I will simply delete some observations.
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)

df[df["Price"] == 0]["StockCode"].unique()
df = df[df["StockCode"] != "M"]

invalid_codes = df[df["StockCode"].astype(str).str.contains(r"[a-zA-Z]{3,}")]["StockCode"].unique().tolist()
df[df["StockCode"].isin(invalid_codes)].groupby(["StockCode"]).agg({"Invoice": "nunique",
                                                                    "Quantity": "sum",
                                                                    "Price": "sum",
                                                                    "Customer ID": "nunique"})
df = df[~df["StockCode"].isin(invalid_codes)].reset_index(drop=True)
check_df(df)


last_invoice_date = df["InvoiceDate"].max()
# Assume a hypothetical day to analyze data: 1 day after.
today_date = (last_invoice_date + dt.timedelta(days=1))

df["NEW_TOTAL_PRICE"] = df["Quantity"] * df["Price"]
RFM = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,  # Recency
                                     "Invoice": lambda x: x.nunique(),  # Frequency
                                     "NEW_TOTAL_PRICE": lambda x: x.sum()})  # Monetary

RFM.columns = ["RECENCY", "FREQUENCY", "MONETARY"]
RFM = RFM[(RFM["MONETARY"]) > 0 & (RFM["FREQUENCY"] > 0)]  # Making sure there is no non-profit observation

check_df(RFM)

check_outlier(RFM, "RECENCY")

for col in RFM.columns:
    print(col, check_outlier(RFM, col))
for col in RFM.columns:
    replace_with_thresholds(RFM, col)
# ------------------------------------------------------#
# DISTRIBUTION BEFORE LOG TRANSFORMATION
RFM["RECENCY"].hist(bins=20)
plt.title("RECENCY")
plt.show()  # Not normally distributed

RFM["FREQUENCY"].hist(bins=20)
plt.title("FREQUENCY")
plt.show()  # Not normally distributed

RFM["MONETARY"].hist(bins=20)
plt.title("MONETARY")
plt.show()  # Not normally distributed
# ------------------------------------------------------#
# DISTRIBUTION AFTER LOG TRANSFORMATION
for col in RFM.columns:
    RFM[f"LOG_{col}"] = np.log1p(RFM[col])

RFM["LOG_RECENCY"].hist(bins=20)
plt.title("LOG_RECENCY")
plt.show()

RFM["LOG_FREQUENCY"].hist(bins=20)
plt.title("LOG_FREQUENCY")
plt.show()

RFM["LOG_MONETARY"].hist(bins=20)
plt.title("LOG_MONETARY")
plt.show()
# ------------------------------------------------------#
# Why scaling?
# LS_: Log-Scaled
sc = StandardScaler()
sc.fit(RFM[["LS_RECENCY", "LS_FREQUENCY", "LS_MONETARY"]])
scaled_rf = sc.transform(RFM[["LS_RECENCY", "LS_FREQUENCY", "LS_MONETARY"]])
new_df = pd.DataFrame(index=RFM.index, columns=["LS_RECENCY", "LS_FREQUENCY", "LS_MONETARY"], data=scaled_rf)
# ------------------------------------------------------#

new_df["LS_RECENCY"].hist(bins=20)
plt.title("LS_RECENCY")
plt.show()

new_df["LS_FREQUENCY"].hist(bins=20)
plt.title("LS_FREQUENCY")
plt.show()

new_df["LS_MONETARY"].hist(bins=20)
plt.title("LS_MONETARY")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(new_df)
elbow.show()

elbow.elbow_value_


#################
# final cluster
#################
def load(RFM):
    RFM = df.groupby("CUSTOMER ID").agg({"INVOICEDATE": lambda x: (last_date - x.max()).days,
                                         "INVOICE": lambda x: x.nunique(),
                                         "NEW_TOTAL_PRICE": lambda x: x.sum()})

    RFM.columns = ["RECENCY", "FREQUENCY", "MONETARY"]
    return RFM


load(RFM)

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=17).fit(new_df)
clusters = kmeans.labels_

new_df["CLUSTER"] = clusters
new_df["CLUSTER"] = new_df["CLUSTER"] + 1
new_df["CLUSTER"].value_counts()

new_df.groupby("CLUSTER").agg(["count", "mean", "median"])

new_df.head()
