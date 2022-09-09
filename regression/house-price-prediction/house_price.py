import sklearn
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
# DATA PREPROCESSING
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor

# MODELING
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning


import xgboost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# MODEL TUNING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# WARNINGS
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_train = pd.read_csv('.../HousePrice/train.csv')
df_test = pd.read_csv('.../HousePrice/test.csv')

df_train.shape, df_test.shape

df = df_train.append(df_test).reset_index(drop=True)
df.shape
df.head()

# "I would recommend removing any houses with more than 4000 square feet from the data set" search in link below.
# ref: http://jse.amstat.org/v19n3/decock.pdf
df_train = df_train[df_train["GrLivArea"] <= 4000]


##### EDA #####

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

sns.displot(df["SalePrice"])


df["SalePrice"].hist(bins=100)
plt.show()
print("Skewness before log: %f" % df['SalePrice'].skew())

np.log1p(df['SalePrice']).hist(bins=100)
plt.show()
print("Skewness after log: %f" % np.log1p(df['SalePrice']).skew())


##### Correlation analysis #####


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)

##### Outlier Analysis #####
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train)


##### Missing Value Analysis #####
# There are a lot of missing values in this dataset. However, we know that those values have information because
# houses simply don't have pool, fireplace, alley, fence etc.

# So we will not drop missing values, we will replace missing values with "No". Therefore, our ML model
# will not raise any error due to missing values.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


no_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "BsmtQual", "BsmtCond", "BsmtExposure",
           "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]

for col in no_cols:
    df[col].fillna("No", inplace=True)


df.shape


def quick_missing_imp(df, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in df.columns if
                         df[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = df[target]

    print("# BEFORE")
    print(df[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                  axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        df = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    df[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(df[variables_with_na].isnull().sum(), "\n\n")

    return df


df = quick_missing_imp(df, num_method="median", cat_length=17)

missing_values_table(df)
##### Feature Engineering #####

# Rare analysis
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# Combine: MSZoning, LotShape, ExterCond, GarageQual, BsmtFinType2, Condition1, BldgType
# Remove: Street, Alley, LandContour, Utilities, LandSlope, Condition2, Heating, CentralAir,
#         Functional, PoolQC, MiscFeature, Neighborhood, KitchenAbvGr
rare_analyser(df, "SalePrice", cat_cols)

New_LotArea = pd.Series(["Studio", "Small", "Middle", "Large", "Dublex", "Luxury"], dtype="category")
df["New_LotArea"] = New_LotArea
df.loc[(df["LotArea"] <= 2000), "New_LotArea"] = New_LotArea[0]
df.loc[(df["LotArea"] > 2000) & (df["LotArea"] <= 4000), "New_LotArea"] = New_LotArea[1]
df.loc[(df["LotArea"] > 4000) & (df["LotArea"] <= 6000), "New_LotArea"] = New_LotArea[2]
df.loc[(df["LotArea"] > 6000) & (df["LotArea"] <= 8000), "New_LotArea"] = New_LotArea[3]
df.loc[(df["LotArea"] > 10000) & (df["LotArea"] <= 12000), "New_LotArea"] = New_LotArea[4]
df.loc[df["LotArea"] > 12000, "New_LotArea"] = New_LotArea[5]
df["New_LotArea"].value_counts()

df["MSZoning"].value_counts()
df.loc[(df["MSZoning"] == "RH"), "MSZoning"] = "RM"
df.loc[(df["MSZoning"] == "FV"), "MSZoning"] = "FV + C (all)"
df.loc[(df["MSZoning"] == "C (all)"), "MSZoning"] = "FV + C (all)"
df["MSZoning"].value_counts()

df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR1"), "LotShape"] = "IR"
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR"
df["LotShape"].value_counts()

df["ExterCond"].value_counts()
df["ExterCond"] = np.where(df["ExterCond"].isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df["ExterCond"].isin(["Ex", "Gd"]), "ExGd", df["ExterCond"])
df['ExterCond'].value_counts()

df['GarageQual'].value_counts()
df["GarageQual"] = np.where(df["GarageQual"].isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df["GarageQual"].isin(["Ex", "Gd"]), "ExGd", df["GarageQual"])
df["GarageQual"] = np.where(df["GarageQual"].isin(["ExGd", "TA"]), "ExGd", df["GarageQual"])
df['GarageQual'].value_counts()

df['BsmtFinType1'].value_counts()
df['BsmtFinType2'].value_counts()
df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType1"])
df["BsmtFinType1"] = np.where(df.BsmtFinType1.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType1"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])
df['BsmtFinType1'].value_counts()
df['BsmtFinType2'].value_counts()

df['Condition1'].value_counts()
df.loc[(df["Condition1"] == "Feedr") | (df["Condition1"] == "Artery") | (df["Condition1"] == "RRAn") | (
        df["Condition1"] == "PosA") | (df["Condition1"] == "RRAe"), "Condition1"] = "AdjacentCondition"
df.loc[(df["Condition1"] == "RRNn") | (df["Condition1"] == "PosN") | (
        df["Condition1"] == "RRNe"), "Condition1"] = "WithinCondition"
df.loc[(df["Condition1"] == "Norm"), "Condition1"] = "NormalCondition"
df['Condition1'].value_counts()

df['BldgType'].value_counts()
df["BldgType"] = np.where(df.BldgType.isin(["1Fam", "2fmCon"]), "Normal", df["BldgType"])
df["BldgType"] = np.where(df.BldgType.isin(["TwnhsE", "Twnhs", "Duplex"]), "Big", df["BldgType"])
df['BldgType'].value_counts()

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual",
                      "GarageCond", "Fence"]].sum(axis=1)

df["Overall"] = df[["OverallQual", "OverallCond"]].sum(axis=1)
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["NEW_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["NEW_PorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["3SsnPorch"] + df["WoodDeckSF"]
df["NEW_TotalHouseArea"] = df["NEW_TotalFlrSF"] + df["TotalBsmtSF"]
df["NEW_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]
df["NEW_TotalFullBath"] = df["BsmtFullBath"] + df["FullBath"]
df["NEW_TotalHalfBath"] = df["BsmtHalfBath"] + df["HalfBath"]
df["NEW_TotalBath"] = df["NEW_TotalFullBath"] + (df["NEW_TotalHalfBath"] * 0.5)
df["NEW_LotRatio"] = df["GrLivArea"] / df["LotArea"]
df["NEW_RatioArea"] = df["NEW_TotalHouseArea"] / df["LotArea"]
df["NEW_GarageLotRatio"] = df["GarageArea"] / df["LotArea"]
df["NEW_Restoration"] = df["YearRemodAdd"] - df["YearBuilt"]
df["NEW_HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["NEW_RestorationAge"] = df["YrSold"] - df["YearRemodAdd"]
df["NEW_GarageAge"] = df["GarageYrBlt"] - df["YearBuilt"]
df["NEW_GarageRestorationAge"] = np.abs(df["GarageYrBlt"] - df["YearRemodAdd"])
df["NEW_GarageSold"] = df["YrSold"] - df["GarageYrBlt"]

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Heating", "PoolQC", "MiscFeature",
             "Neighborhood", "KitchenAbvGr", "CentralAir", "Functional"]

df.drop(drop_list, axis=1, inplace=True)

df.shape
##### Encoding #####

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape

##### Modeling #####
missing_values_table(df)

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

train_df.shape, test_df.shape

y = np.log1p(df[df['SalePrice'].notnull()]['SalePrice'])  # null values of "SalePrice" comes from test dataset.
X = train_df.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


def all_models(X, y, test_size=0.2, random_state=42, classification=False):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor())]
        # ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df


all_models = all_models(X_train, y_train, test_size=0.2, random_state=42, classification=False)

##### Hyperparameter Optimization #####

# At this point we can see lowest RMSE_Test is Ridge. However, this does not mean Ridge is the best model for us.
# RMSE_Test wise Ridge is the best model but, do we need 1-2% more accurate model for $10k more cost?

ridge_model = Ridge(random_state=42).fit(X_train, y_train)
ridge_model.get_params()
ridge_params = {'alpha':[1, 0.5, 0.1, 0.01, 1.5]}
ridge_best_grid = GridSearchCV(ridge_model, ridge_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
ridge_best_grid.best_params_
ridge_final_model = ridge_model.set_params(**ridge_best_grid.best_params_).fit(X, y)
y_train_pred = ridge_final_model.predict(X_train)
y_test_pred = ridge_final_model.predict(X_test)

np.sqrt(mean_squared_error(y_train, y_train_pred))
np.sqrt(mean_squared_error(y_test, y_test_pred))

ridge_rmse = np.mean(np.sqrt(-cross_val_score(ridge_final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

np.expm1(ridge_rmse)
## GBM ##
gbm_model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)

gbm_model.get_params()

gbm_params = {"learning_rate": [0.1, 0.03],
              "max_depth": [3, 5, 6],
              "n_estimators": [100, 200, 300],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)

gbm_best_grid.best_params_

final_model = gbm_model.set_params(**gbm_best_grid.best_params_).fit(X, y)

y_train_pred = final_model.predict(X_train)

print("GBM Tuned Model Train RMSE:", round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4))

y_test_pred = final_model.predict(X_test)
print("GBM Tuned Model Test RMSE:", round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4))

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

y_pred_train = np.expm1(y_train_pred)
y_pred_test = np.expm1(y_test_pred)

y_train_inv = np.expm1(y_train)
y_test_inv = np.expm1(y_test)

np.sqrt(mean_squared_error(y_pred_test, y_test_inv))
###################################################

## XGBoost ##
xgboost_model = XGBRegressor(objective='reg:squarederror')

xgboost_params = {"learning_rate": [0.1, 0.01, 0.03],
                  "max_depth": [5, 6, 8],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1]}

xgboost_gs_best = GridSearchCV(xgboost_model,
                            xgboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = xgboost_model.set_params(**xgboost_gs_best.best_params_).fit(X, y)

xgboost_gs_best.best_params_

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

rmse_train = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
rmse_test = np.mean(np.sqrt(-cross_val_score(final_model, X_test, y_test, cv=5, scoring="neg_mean_squared_error")))


##### Plot importance for GBM #####
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X)

