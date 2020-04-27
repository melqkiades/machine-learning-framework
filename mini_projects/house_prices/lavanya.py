
# Essentials
import numpy
import pandas
import datetime
import random
import time

# Plots
import seaborn
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
# import lightgbm as lgb
# from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pandas.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pandas.options.display.max_seq_items = 8000
pandas.options.display.max_rows = 8000


# Taken from: https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition


TRAIN_FILE = '/Users/fpena/Datasets/house-prices/train.csv'
TEST_FILE = '/Users/fpena/Datasets/house-prices/test.csv'
SAMPLE_SUBMISSION_FILE = '/Users/fpena/Datasets/house-prices/sample_submission.csv'
SUBMISSION_FILE_1 = '/Users/fpena/Datasets/house-prices/submission_regression1.csv'
SUBMISSION_FILE_2 = '/Users/fpena/Datasets/house-prices/submission_regression2.csv'

# start = time.time()



# Read in the dataset as a dataframe

def load_data():
    train = pandas.read_csv(TRAIN_FILE)
    test = pandas.read_csv(TEST_FILE)

    return train, test


def rescale_sale_price(train):

    train["SalePrice"] = numpy.log1p(train["SalePrice"])
    return train



# Remove outliers
def remove_outliers(train):
    train.drop(train[(train['OverallQual' ] <5) & (train['SalePrice' ] >200000)].index, inplace=True)
    train.drop(train[(train['GrLivArea' ] >4500) & (train['SalePrice' ] <300000)].index, inplace=True)
    train.reset_index(drop=True, inplace=True)

    return train



def split_features(train, test):

    # Split features and labels
    train_labels = train['SalePrice'].reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test

    # Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
    all_features = pandas.concat([train_features, test_features]).reset_index(drop=True)

    # Some of the non-numeric predictors are stored as numbers; convert them into strings
    all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
    all_features['YrSold'] = all_features['YrSold'].astype(str)
    all_features['MoSold'] = all_features['MoSold'].astype(str)

    all_features = handle_missing(all_features)

    # missing = percent_missing(all_features)
    # df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
    # print('Percent of missing data')
    # df_miss[0:10]

    # Let's make sure we handled all the missing values
    # missing = percent_missing(all_features)
    # df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

    return all_features, train_labels


# determine the threshold for missing values
def percent_missing(df):
    data = pandas.DataFrame(df)
    df_cols = list(pandas.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean() * 100, 2)})

    return dict_x


def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))

    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))
    return features



def normalize_numeric_features(all_features):

    # Fetch all numeric features
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in all_features.columns:
        if all_features[i].dtype in numeric_dtypes:
            numeric.append(i)



    # Find skewed numerical features
    skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index

    print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
    skewness = pandas.DataFrame({'Skew' :high_skew})

    # Normalize skewed features
    for i in skew_index:
        all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))

    all_features['BsmtFinType1_Unf'] = 1 * (all_features['BsmtFinType1'] == 'Unf')
    all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
    all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
    all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
    all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
    all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
    all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
    all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
    all_features = all_features.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)
    all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
    all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

    all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                         all_features['1stFlrSF'] + all_features['2ndFlrSF'])
    all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                                       all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
    all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                                      all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                                      all_features['WoodDeckSF'])
    all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: numpy.exp(6) if x <= 0.0 else x)
    all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: numpy.exp(6.5) if x <= 0.0 else x)
    all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: numpy.exp(6) if x <= 0.0 else x)
    all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: numpy.exp(4.2) if x <= 0.0 else x)
    all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: numpy.exp(4) if x <= 0.0 else x)
    all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: numpy.exp(6.5) if x <= 0.0 else x)

    all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    log_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YearRemodAdd', 'TotalSF']

    all_features = logs(all_features, log_features)

    squared_features = ['YearRemodAdd', 'LotFrontage_log',
                        'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
                        'GarageCars_log', 'GarageArea_log']
    all_features = squares(all_features, squared_features)

    all_features = pandas.get_dummies(all_features).reset_index(drop=True)

    # Remove any duplicated column names
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    return all_features


def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pandas.Series(numpy.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pandas.Series(res[l] * res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res


def create_x_sets(all_features, train_labels):
    X = all_features.iloc[:len(train_labels), :]
    X_test = all_features.iloc[len(train_labels):, :]

    # Finding numeric features
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in X.columns:
        if X[i].dtype in numeric_dtypes:
            if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasgarage', 'hasbsmt', 'hasfireplace']:
                pass
            else:
                numeric.append(i)

    return X, X_test


###############
# Train model #
###############


# Define error metrics
def rmsle(y, y_pred):
    return numpy.sqrt(mean_squared_error(y, y_pred))


def cv_rmse(model, X, train_labels):
    # Setup cross validation folds
    kf = KFold(n_splits=12, random_state=42, shuffle=True)
    rmse = numpy.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return rmse


def train_models(X, train_labels):

    # Light Gradient Boosting Regressor
    # lightgbm = LGBMRegressor(objective='regression',
    #                        num_leaves=6,
    #                        learning_rate=0.01,
    #                        n_estimators=7000,
    #                        max_bin=200,
    #                        bagging_fraction=0.8,
    #                        bagging_freq=4,
    #                        bagging_seed=8,
    #                        feature_fraction=0.2,
    #                        feature_fraction_seed=8,
    #                        min_sum_hessian_in_leaf = 11,
    #                        verbose=-1,
    #                        random_state=42)

    # XGBoost Regressor
    xgboost = XGBRegressor(learning_rate=0.01,
                           n_estimators=6000,
                           max_depth=4,
                           min_child_weight=0,
                           gamma=0.6,
                           subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:linear',
                           nthread=-1,
                           scale_pos_weight=1,
                           seed=27,
                           reg_alpha=0.00006,
                           random_state=42)

    # Ridge Regressor
    ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20,
                    30, 50, 75, 100]
    kfold = KFold(n_splits=12, random_state=42, shuffle=True)
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kfold))

    # Support Vector Regressor
    svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=6000,
                                    learning_rate=0.01,
                                    max_depth=4,
                                    max_features='sqrt',
                                    min_samples_leaf=15,
                                    min_samples_split=10,
                                    loss='huber',
                                    random_state=42)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=1200,
                               max_depth=15,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               max_features=None,
                               oob_score=True,
                               random_state=42)

    # Stack up all the models above, optimized using xgboost
    stack_gen = StackingCVRegressor(regressors=(xgboost, svr, ridge, gbr, rf),
                                    meta_regressor=xgboost,
                                    use_features_in_secondary=True)

    scores = {}

    # score = cv_rmse(lightgbm)
    # print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    # scores['lgb'] = (score.mean(), score.std())

    score = cv_rmse(xgboost, X, train_labels)
    print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['xgb'] = (score.mean(), score.std())

    score = cv_rmse(svr, X, train_labels)
    print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['svr'] = (score.mean(), score.std())

    score = cv_rmse(ridge, X, train_labels)
    print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['ridge'] = (score.mean(), score.std())

    score = cv_rmse(rf, X, train_labels)
    print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['rf'] = (score.mean(), score.std())

    score = cv_rmse(gbr, X, train_labels)
    print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['gbr'] = (score.mean(), score.std())

    print('stack_gen')
    stack_gen_model = stack_gen.fit(numpy.array(X), numpy.array(train_labels))

    # print('lightgbm')
    # lgb_model_full_data = lightgbm.fit(X, train_labels)

    print('xgboost')
    xgb_model_full_data = xgboost.fit(X, train_labels)

    print('Svr')
    svr_model_full_data = svr.fit(X, train_labels)

    print('Ridge')
    ridge_model_full_data = ridge.fit(X, train_labels)

    print('RandomForest')
    rf_model_full_data = rf.fit(X, train_labels)

    print('GradientBoosting')
    gbr_model_full_data = gbr.fit(X, train_labels)

    # Blend models in order to make the final predictions more robust to overfitting
    blended_predictions_train = (
        (0.12 * ridge_model_full_data.predict(X)) +
        (0.22 * svr_model_full_data.predict(X)) +
        (0.12 * gbr_model_full_data.predict(X)) +
        (0.12 * xgb_model_full_data.predict(X)) +
        # (0.1 * lgb_model_full_data.predict(X)) +
        (0.07 * rf_model_full_data.predict(X)) +
        (0.35 * stack_gen_model.predict(numpy.array(X))))

    # Get final precitions from the blended model
    blended_score_train = rmsle(train_labels, blended_predictions_train)
    scores['blended'] = (blended_score_train, 0)
    print('RMSLE score on train data:')
    print(blended_score_train)

    return ridge_model_full_data, svr_model_full_data, gbr_model_full_data, xgb_model_full_data, rf_model_full_data,\
           stack_gen_model


def prepare_submission(
        X_test,
        ridge_model_full_data, svr_model_full_data, gbr_model_full_data, xgb_model_full_data, rf_model_full_data,
        stack_gen_model):

    submission = pandas.read_csv(SAMPLE_SUBMISSION_FILE)

    blended_predictions_test = (
            (0.12 * ridge_model_full_data.predict(X_test)) +
            (0.22 * svr_model_full_data.predict(X_test)) +
            (0.12 * gbr_model_full_data.predict(X_test)) +
            (0.12 * xgb_model_full_data.predict(X_test)) +
            # (0.1 * lgb_model_full_data.predict(X_test)) +
            (0.07 * rf_model_full_data.predict(X_test)) +
            (0.35 * stack_gen_model.predict(numpy.array(X_test))))

    # Append predictions from blended models
    submission.iloc[:,1] = numpy.floor(numpy.expm1(blended_predictions_test))

    # Fix outleir predictions
    q1 = submission['SalePrice'].quantile(0.0045)
    q2 = submission['SalePrice'].quantile(0.99)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
    submission.to_csv("submission_regression1.csv", index=False)
    # Score 0.11987

    # Scale predictions
    submission['SalePrice'] *= 1.001619
    submission.to_csv("submission_regression2.csv", index=False)
    # Score 0.11979


def full_cycle():
    train, test = load_data()
    train = rescale_sale_price(train)
    train = remove_outliers(train)
    all_features, train_labels = split_features(train, test)
    all_features = normalize_numeric_features(all_features)
    X, X_test = create_x_sets(all_features, train_labels)
    ridge_model, svr_model, gbr_model, xgb_model, rf_model, stack_gen_model = train_models(X, train_labels)
    prepare_submission(X_test, ridge_model, svr_model, gbr_model, xgb_model, rf_model, stack_gen_model)


# TODO: Plant random seeds

start = time.time()
full_cycle()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
