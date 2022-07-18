#############################################################################################
# Python ile veri analizi
# numpy
# pandas
# veri görselleştirme
# gelişmiş foknksiyonel keşifçi veri analizi (advanced functional exploratory data analysis)
#############################################################################################
# sabit tipte veri tutar, hızlıdır, high-level (vektörel) seviyede işlem yapar. az çaba, çok işlem.
import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(len(a)):
    ab.append(a[i] * b[i])

# or
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#################
# numpy array oluşturmak
#################
#import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.randint(0, 10, size=20)
np.random.normal(10, 4, (3, 4)) #avg: 10, variance:4, 3x4

##################
# numpy array özellikleri (attributes)
##################
# import numpy as np

a = np.random.randint(10, size=5)
a.ndim # boyut sayısı
a.shape # boyut bilgisi
a.size # toplam eleman sayısı
a.dtype # tip bilgisi, int32, int64 vs.

################
# reshaping
###############
# import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3 ,3)

# or
ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

########################
#index seçimi (selection)
########################
# import numpy as np
a = np.random.randint(10, size=10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3, 5)) #3x5'lik 0-10 arasında
m[0, 0]
m[2, 1] = 999
m[2, 1] = 2.9
m[:, 0]
m[1, :]
m[0:2, 0:3]

######################
#fancy index < bil.
######################
# import numpy as np

v = np.arange(0, 30, 3)
v[1]
v[2]
v[3]

#or
catch = [1, 2, 3]
v[catch]

#######################
#conditions
#######################
# import numpy as np
v = np.array([1, 2, 3, 4, 5])

# klasik döngü ile
ab = []
for i in v:
    if i <3:
        ab.append(i)

# numpy ile
v < 3
v[v < 3] #true olanları seçecek, false olanları seçmicek
v[v > 3]
v[v != 3]

#####################
# matematiksel işlemler
#####################
# import numpy as np
v = np.array([1, 2, 3, 4, 5])
v / 5
v * 5 / 10
v ** 2
v -1

np.subtract(v, 1) # v'nin bütün elemanlarından 1 çıkardı
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

#################
# np ile 2 bilinmeytenli denklem çözümü
#################
# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])
np.linalg.solve(a, b)

##################################################################
# PANDAS
##################################################################

##########
# pandas series
##########
import pandas as pd

s = pd.Series([10, 66, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3) # ilk 3 eleman
s.tail(3) # son 3 eleman

############
#veri okuma (data reading)
############

# import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head() # ilk 5 elemanı oku

############
# veriye hızlı bakış (quick look at data)
############
# import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any() # eksiklik var mi
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()  # SADECE KATEGORİK DEĞİŞKENLER İÇİN !!!!!!!!!!!!!11

##################
# Pandas'ta seçim işlemleri ( selection in pandas)
##################
# import pandas as pd
# import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df[0:13]
df.drop(0 ,axis=0).head() # axis=0 row, axis=1 column

delete_indexes = [1 ,3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# kalıcı olarak silmek için
# df = df.drop(delete_indexes, axis=0)
# veya
# df.drop(delete_indexes, axis=0, inplace=True)

############################
# değişkeni indexe çevirmek
############################

df["age"].head()
#veya
df.age.head()

df.index = df["age"] # değişkeni indexe at

df.drop("age", axis=1).head() # kolon sil

df.drop("age", axis=1, inplace=True) # değişkeni sil
df.head()

# df.drop("index", axis=1, inplace=True)
###############
# indexi değişkene çevirmek
###############

df["age"] = df.index # indexi değişkene at
df.head()

# 2. yol
df = df.reset_index().head() #indexi sil

###############
# değişkenler üzerinde işlemler
###############

# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None) # bütün değişkenleri göster
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
#or
df.age.head()

type(df["age"].head()) # pandas series
type(df[["age"]].head()) # dataframe

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()
df.drop(col_names, axis=1).head() #birden fazla değişken silmek

# str.contains sık kullanılır.
df.loc[:, df.columns.str.contains("age")].head() # değişkenlerde "yaş" ifadesi olanları seç
df.loc[:, ~df.columns.str.contains("age")].head() # "yaş" olmayanları seç

##############
# iloc & loc
# çok sık kullanılır
##############

# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3] # 0'dan 3'e kadar
df.iloc[0, 0]

# loc: label based selection
df.loc[0:3] # 3. index de dahil
####

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

######################
# koşullu seçim (conditional selection)
#####################

# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head() # koşulu sağlayan bütün sütunları getir
df[df["age"] > 50]["age"] # koşulu sağlayan age sütununu getir

df[df["age"] > 50]["age"].count() # koşulu sağlayan age sütun sayısını getir

df.loc[df["age"] > 50, ["age", "class"]].head() # koşulu sağlayan age ve class sütunlarını getir

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head() # iki koşulu da sağlayan age ve class getir

df["embark_town"].value_counts()

# yaşları 50'den büyük, erkek, cherbourg veya southampton'dan gelenlerin
# age, class ve embark_town sütunlarını getir.
df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

############################
#toplulaştırma ve gruplama (aggregation & grouping)
############################
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

# cinsiyet bazında yaş ortalaması
# Groupby: kırılım, tekilleştirme
df.groupby("sex")["age"].mean() # cinsyete göre grupla, yaş ortalamasını al

df.groupby("sex").agg({"age": "mean"}) # bu kullanımı öğren
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})

######################
# Pivot table
######################

# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

#yaşa ve gemiye bindikleri limana göre hayatta kalma oranları
df.pivot_table("survived", "sex", "embarked") # "kesişim", "satır", "sütun"

#yaşa ve gemiye bindikleri limana ve hangi sınıftan bilet aldıklarına göre hayatta kalma oranları
df.pivot_table("survived", "sex", ["embarked", "class"])

# NUMERİK DEĞERİ KATEGORİZE ETMEK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option("display.width", 500)

########################
# apply ve lambda
########################
# import pandas as pd
# import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head() # olay bu

# ya da

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler()).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler()).head()

#############################
# birleştirme (join) işlemleri
#############################
# import pandas as pd
# import seaborn as sns
import numpy as np
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2]) #liste içerisinde
pd.concat([df1, df2], ignore_index=True)

###################
# merge ile birleştirme işlemleri
###################

df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group" : ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees") #<< iki ifade de aynı

# amaç: her çalışanın müdürünün bilgisine erişmek istiyoruz.

df3 = pd.merge(df1, df2)
df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                    "manager": ["caner", "musto"," serkcan"]})
pd.merge(df3, df4)
pd.merge(df3, df4, on="group") # iki ifade de aynı

#################################
# veri görselleştirme: matplotlib & seaborn
#################################

#############
# matplotlib
#############

# kategorik değişken için: sütun grafik. countplot bar
# sayısal değişken için: histogram, boxplot

########################
# kategorik değişken görselleştirme
########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

######################
# sayısal değişken görselleştirme
######################
plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

###############################
# matplotlib'in özellikleri
###############################

#########
# plot
#########
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

x = np.array([1, 3, 5, 7, 9])
y = np.array([2, 4, 6, 8, 10])
plt.plot(x, y, "o")
plt.show()


##############
# marker
##############
y = np.array([13, 28, 11, 100])
plt.plot(y, marker="o")
plt.show()

plt.plot(y, marker="*")
plt.show()

############
# line
############
y = np.array([13, 28, 11, 100])
plt.plot(y)
plt.show()

plt.plot(y, linestyle="dotted")
plt.show()

#############
# multiple lines
#############
x = np.array([1, 3, 5, 7, 9])
y = np.array([2, 4, 6, 8, 10])
plt.plot(x)
plt.plot(y)
plt.show()

##############
# labels
##############
x = np.array([1, 3, 5, 7, 9])
y = np.array([2, 4, 6, 8, 10])
plt.plot(x, y)

plt.title("ana başlık")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid()
plt.show()

############
# subplot
############
x = np.array([1, 3, 5, 7, 9])
y = np.array([2, 4, 6, 8, 10])
plt.subplot(1, 2, 1) #1'e 2'lik grafiğin 1.si
plt.title("1")
plt.plot(x, y)

x = np.array([11, 31, 51, 71, 91])
y = np.array([22, 42, 62, 82, 102])
plt.subplot(1, 2, 2)
plt.title("2") #1'e 2'lik grafiğin 2.si
plt.plot(x, y)

####################################################################
# SEABORN
####################################################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

##################################
# kategorik değişken görselleştirme
##################################

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

##################################
# sayısal değişken görselleştirme
##################################
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ ( ADVANCED FUNCTIONAL EDA)
################################################################
# 1. genel resim
# 2. kategorik değişken analizi (categorical var. analysis)
# 3. sayısal değişken analizi ( numerical var. analysis)
# 4. hedef değişken analizi (target var. analysis)
# 5. korelasyon analizi (correlation analysis)

###################################
# 1. genel resim
###################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5): # Senior'lar çok sever bu fonksiyonu
    print("################ Shape ##################")
    print(dataframe.shape)
    print("################ Types ##################")
    print(dataframe.dtypes)
    print("################ Head ##################")
    print(dataframe.head(head))
    print("################ Tail ##################")
    print(dataframe.tail(head))
    print("################ NA ##################")
    print(dataframe.isnull().sum())
    print("################ Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
    print("#################### END ###################")

check_df(df)

df = sns.load_dataset("flights")
check_df(df)

###################################
# 2. kategorik değişken analizi
###################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique() #içindeki sınıf isimleri
df["sex"].nunique() #sınıf sayısı (numeric unique)


# ÇOK KRİTİK
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_bat_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes in ["category", "object"]]

cat_cols = cat_cols + num_bat_cat

cat_cols = [col for col in df.columns if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)

# Sektörde:
def cat_summary(dataframe, col_name):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)
########################################################
def cat_summary(dataframe, col_name, plot=False):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df,"sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("fgddhfahhahf")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int) # normalde bool, astype yapınca, true gördükleri 1, false gördükleri 0.
################################################################################################
for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# bu alttaki çok boktan bi kod.
def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print("######################################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("######################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print("######################################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("######################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df,"adult_male", plot=True)


# özetle en verimlisi: do one thing.
def cat_summary(dataframe, col_name):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")


#######################################################################
# 3. sayısal değişken analizi
#######################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age", "fare"]].describe().T
df[["age", "fare"]].dtypes
num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

##########################################################################
# değişkenlerin yakalanması ve işlemlerin genelleştirilmesi  <<<<<< ÇOK ÖNEMLİ
##########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# docstring
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişjkenler için sınıf eşik değeri
    Returns
    -------
        cat_cols: list
            Kategorik değişken listesi
        num_cols: list
            Numnerik değişken listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    cat_cols = [ col for col in df.columns if str(df[col].dtypes) in ["category", "bool", "object"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in df.columns if col not in cat_cols]

    print(f"Observations : {dataframe.shape[0]}") # satır sayısı
    print(f"Variables : {dataframe.shape[1]}") # değişken sayısı
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##### özet ######

# kategorik için
def cat_summary(dataframe, col_name):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")

cat_summary(df, "sex") # birini dene

for col in cat_cols:
    cat_summary(df, col)

# numerik için
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# bonus
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")
cat_cols, num_cols, cat_but_car = grab_col_names(df)

###################################################################
# 4. hedef değişken analizi
###################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype("int64")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişjkenler için sınıf eşik değeri
    Returns
    -------
        cat_cols: list
            Kategorik değişken listesi
        num_cols: list
            Numnerik değişken listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    cat_cols = [ col for col in df.columns if str(df[col].dtypes) in ["category", "bool", "object"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations : {dataframe.shape[0]}") # satır sayısı
    print(f"Variables : {dataframe.shape[1]}") # değişken sayısı
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

def cat_summary(dataframe, col_name):
    print("######################################")
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################################")
df["survived"].value_counts()
cat_summary(df, "survived")

####
#hedef değişkenin kategorik değişkenlerle ile analizi
####

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_col)[target].mean()}))

target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    print("#################")
    target_summary_with_cat(df, "survived", col)
    print("#################")

### ikisi de aynı, alttakinin çıktısı daha okunaklı olduğu için fonksiyonda kullanılabilir.
df.groupby("survived")["age"].mean()
#or
df.groupby("survived").agg({"age": "mean"})
###

def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}))

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#####################################################################################################
# 5. korelasyon analizi
#####################################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 1:-1] # 1. ve sonuncu değişkenleri salla (problemi değişkenler)
df.head()
df.info()
# sadece bir analiz aracı olarak kullanmalısınız.

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
plt.interactive(False)

#####################################################
# Yüksek korelasyonlu değişkenlerin silinmesi
#####################################################
# korelasyonu çok yüksek (0.99) olan değişkenlerin silinmesi gerekir çünkü neredeyse
# birbirlerinin aynılarıdır.

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.9)] # yüksek corr.

cor_matrix[drop_list] # yüksek korelasyonluları görelim
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()

    return drop_list

drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
drop_list = high_correlated_cols(df.drop(drop_list, axis=1), plot=True)