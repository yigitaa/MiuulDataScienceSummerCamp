import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 1000)

import warnings

warnings.filterwarnings("ignore")

##### PostgreSQL connection #####

user = "postgres"
password = "1231"
host = "localhost"
port = "5432"
database = "olist"

connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
con = create_engine(connection_string)

df_customers = pd.read_sql('select * from customers', con)
df_geolocation = pd.read_sql('select * from geo_location', con)
df_orders = pd.read_sql('select * from orders', con)
df_order_items = pd.read_sql('select * from order_items', con)
df_order_payments = pd.read_sql('select * from order_payments', con)
df_order_reviews = pd.read_sql('select * from order_reviews', con)
df_products = pd.read_sql('select * from products', con)
df_sellers = pd.read_sql('select * from sellers', con)
df_translations = pd.read_sql('select * from product_translation', con)

df_list_col = ['df_customers', 'df_geolocation', 'df_orders', 'df_order_items', 'df_order_payments',
               'df_order_reviews', 'df_products', 'df_sellers', 'df_translations']

df_list_col.shape


# easy to read one by one.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    # print("##################### Tail #####################")
    # print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df_customers)  # ok
check_df(df_geolocation)  # ok
check_df(df_orders)  # missing values
check_df(df_order_items)  # ok
check_df(df_order_payments)  # ok
check_df(df_order_reviews)  # missing values
check_df(df_products)  # missing values
check_df(df_sellers)  # ok
check_df(df_translations)  # ok

##### Missing values #####

# missing values in: df_orders
check_df(df_orders)

df_orders['order_approved'].fillna(df_orders['order_purchase'], inplace=True)
df_orders['order_delivered_carrier'].fillna(df_orders['order_approved'], inplace=True)
df_orders['order_delivered_customer'].fillna(df_orders['order_estimated_delivery'], inplace=True)
df_orders.isnull().sum()

# missing values in: df_order_reviews
check_df(df_order_reviews)

df_order_reviews = df_order_reviews[~(df_order_reviews['review_title'].isnull())].reset_index(drop=True)
df_order_reviews['review_comment'] = df_order_reviews['review_comment'].replace(np.nan, 'None')
df_order_reviews.isnull().sum()

# missing values in: df_products
check_df(df_products)

df_products_na_list = ['product_category', 'product_name_length', 'product_desc_length', 'product_photos_qty',
                       'product_weight_grams', 'product_length_cm', 'product_height_cm', 'product_width_cm']

for col in df_products_na_list:
    df_products = df_products[~(df_products[col].isnull())]

df_products.isnull().sum()

##### Merge dataframes #####

df_merge = pd.merge(df_orders, df_order_payments, on='order_id')
df_merge = pd.merge(df_merge, df_customers, on='customer_id')
df_merge = pd.merge(df_merge, df_order_items, on='order_id')
df_merge = pd.merge(df_merge, df_sellers, on='seller_id')
df_merge = pd.merge(df_merge, df_order_reviews, on='order_id')
df_merge = pd.merge(df_merge, df_products, on='product_id')

df_merge.shape
#####
df_translations['product_category'] = df_translations['category']
df_translations.drop('category', axis=1, inplace=True)
# or
df_translations.columns = ['product_category', 'category_translation']
#####
df_merge = pd.merge(df_merge, df_translations, on='product_category')

df_merge.head()
df_merge.info
df_merge.shape
df_merge.isnull().sum()
##### Delete duplications #####


df_merge[df_merge.duplicated(subset={'order_id',
                                     'customer_id',
                                     'order_purchase',
                                     'order_delivered_customer'}, keep='first')].head(30)

df_merge = df_merge.drop_duplicates(subset={'order_id',
                                            'customer_id',
                                            'order_purchase',
                                            'order_delivered_customer'}, keep='first')
df_merge.shape

##### Date conversion #####
df_merge['order_purchase'] = pd.to_datetime(df_merge['order_purchase'],
                                            infer_datetime_format=True,
                                            errors='ignore')

df_merge['order_purchase'].head()

##### Unique number of products #####

df_merge['product_category'].nunique()  # 66 different products

##### Bar-plot #####
df_merge.groupby(['product_category']).agg({'product_id': 'count'}). \
    sort_values(by=['product_id'], ascending=False).head(10)

top10_sold_products = df_merge.groupby('product_category')['product_id'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top10_sold_products.index, y=top10_sold_products.values)
plt.xticks(rotation=80)
plt.xlabel('Product category')
plt.title('Top 10 products')
plt.show()

df_translations.head()

################################################# RFM #################################################


# Monetary
df_merge['TOTAL_SALES_QUANTITY'] = df_merge['payment_value'] * df_merge['payment_installments']

df_monetary = df_merge.groupby(['customer_unique_id'],
                               group_keys=False,
                               as_index=False).agg({'TOTAL_SALES_QUANTITY': 'sum'}).reset_index(drop=True)

df_monetary.rename(columns={'TOTAL_SALES_QUANTITY': 'monetary'}, inplace=True)
df_monetary.head()

# Frequency

df_frequency = df_merge.groupby(['customer_unique_id'],
                                group_keys=False,
                                as_index=False).agg({'order_id': 'count'}).reset_index(drop=True)

df_frequency.rename(columns={'order_id': 'frequency'}, inplace=True)

df_frequency.head()

# Merge monetary and frequency

df = pd.merge(left=df_monetary,
              right=df_frequency,
              on='customer_unique_id',
              how='inner')

df.head()

# Recency
df_merge['order_purchase'].max()

last_date = dt.datetime(2018, 8, 30)

df_merge['DAYS'] = (last_date - df_merge['order_purchase']).dt.days

df_recency = df_merge.groupby(['customer_unique_id'],
                              group_keys=False,
                              as_index=False).agg({'DAYS': 'min'}).reset_index(drop=True)

df_recency.columns = ['customer_unique_id', 'recency']

##### Merge RFM #####

rfm = pd.merge(left=df,
               right=df_recency,
               on='customer_unique_id',
               how='inner')

rfm.head()

##### Plot RFM #####

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1);
sns.distplot(rfm['recency'])
plt.subplot(3, 1, 2);
sns.distplot(rfm['frequency'])
plt.subplot(3, 1, 3);
sns.distplot(rfm['monetary'])
plt.show()

################################################# K-Means Segmentation #################################################

# Standardization

rfm_scaler = rfm[['monetary', 'frequency', 'recency']]
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(rfm_scaler)
model_df = pd.DataFrame(model_scaling, columns=rfm_scaler.columns)
model_df.head()

# Optimal number of cluster: Elbow
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# K-means
k_means = KMeans(n_clusters=5, random_state=42).fit(model_df)
segments = k_means.labels_
rfm['segment'] = segments

# Final

final_merge = df_merge[['customer_unique_id',
                        'order_id', 'order_status', 'order_purchase', 'payment_type', 'payment_installments',
                        'payment_value']]

final_df = pd.merge(left=final_merge,
                    right=rfm,
                    on='customer_unique_id',
                    how='inner')

final_df.groupby('segment').agg({'payment_installments': ['median', 'min', 'max'],
                                 'payment_value': ['median', 'min', 'max'],
                                 'monetary': ['median', 'min', 'max'],
                                 'frequency': ['median', 'min', 'max'],
                                 'recency': ['median', 'min', 'max']})

final_df.head()
