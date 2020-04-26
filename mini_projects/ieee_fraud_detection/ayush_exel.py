from sklearn import metrics, naive_bayes

import numpy
import pandas
import xgboost


# From: https://app.wandb.ai/cayush/kaggle-fraud-detection/reports/Using-W%26B-in-a-Kaggle-Competition--Vmlldzo3MDY2NA
# From: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
import time

import wandb
from sklearn.model_selection import train_test_split

TRAIN_TRANSACTION_FILE = '/Users/fpena/Datasets/ieee-fraud-detection/train_transaction.csv'
TRAIN_IDENTITY_FILE = '/Users/fpena/Datasets/ieee-fraud-detection/train_identity.csv'


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            IsInt = False

            mx = props[col].max()
            mn = props[col].min()

            if not numpy.isfinite(props[col]).all():
                props[col].fillna(-999, inplace=True)

            asint = props[col].fillna(0).astype(numpy.int64)
            result = (props[col] - asint)
            result = result.sum()

            if -0.01 < result < 0.01:
                IsInt = True

            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(numpy.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(numpy.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(numpy.uint32)
                    else:
                        props[col] = props[col].astype(numpy.uint64)
                else:
                    if mn > numpy.iinfo(numpy.int8).min and mx < numpy.iinfo(numpy.int8).max:
                        props[col] = props[col].astype(numpy.int8)
                    elif mn > numpy.iinfo(numpy.int16).min and mx < numpy.iinfo(numpy.int16).max:
                        props[col] = props[col].astype(numpy.int16)
                    elif mn > numpy.iinfo(numpy.int32).min and mx < numpy.iinfo(numpy.int32).max:
                        props[col] = props[col].astype(numpy.int32)
                    elif mn > numpy.iinfo(numpy.int64).min and mx < numpy.iinfo(numpy.int64).max:
                        props[col] = props[col].astype(numpy.int64)
            else:
                props[col] = props[col].astype(numpy.float32)

    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


def load_csv(path):
    return reduce_mem_usage(pandas.read_csv(path))



def load_data():
    train_transaction = load_csv(TRAIN_TRANSACTION_FILE)
    train_identity = load_csv(TRAIN_IDENTITY_FILE)

    trainset = pandas.merge(
        train_transaction, train_identity, on='TransactionID', how='outer')

    return trainset


def transform_time_columns(trainset):

    trainset = trainset[
        ["TransactionID", "DeviceType", "DeviceInfo", "TransactionDT",
         "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
         "R_emaildomain", "addr1", "addr2", "dist1", "dist2", "isFraud"]]

    trainset.groupby('isFraud').agg([numpy.mean, numpy.median])

    trainset['TransactionDT_day'] = trainset['TransactionDT'].apply(
        lambda x: int(x / 86400))
    trainset['TransactionDT_hour'] = trainset['TransactionDT'].apply(
        lambda x: int(x / 86400 % 1 * 24))
    trainset['TransactionDT_min'] = trainset['TransactionDT'].apply(
        lambda x: int(x / 86400 % 1 * 24 % 1 * 60))
    trainset['TransactionDT_sec'] = trainset['TransactionDT'].apply(
        lambda x: int(x / 86400 % 1 * 24 % 1 * 60 % 1 * 60))

    return trainset


#################
# Data cleaning #
#################
def clean_data(trainset):
    # Create a copy of trainset for easy resetting
    dataset = trainset.copy()

    # Replace columns that have empty value with 'unknown' value
    cols = ['card4', 'card6', 'DeviceType', 'DeviceInfo', 'P_emaildomain',
            'R_emaildomain']
    dataset[cols] = dataset[cols].replace({'': 'unknown'})

    # Keep the top 5 column values and group remaining ones into 'Others'
    top5_deviceinfo = set(dataset['DeviceInfo'].value_counts()[:5].index)
    dataset['DeviceInfo'] = dataset['DeviceInfo'].apply(
        lambda x: x if x in top5_deviceinfo else 'Others')

    # Group categories that are similar into one category [4]
    regex_patterns = {
        r'^frontier.*$': 'frontier.com',
        r'^gmail.*$': 'gmail.com',
        r'^hotmail.*$': 'hotmail.com',
        r'^live.*$': 'live.com',
        r'^netzero.*$': 'netzero.com',
        r'^outlook.*$': 'outlook.com',
        r'^yahoo.*$': 'yahoo.com'
    }
    replacements = {
        'P_emaildomain': regex_patterns,
        'R_emaildomain': regex_patterns
    }

    dataset.replace(replacements, regex=True, inplace=True)

    return dataset


#################
# Preprocessing #
#################
def preprocessing(dataset):
    # Use sine and cosine for time of the day as these are cyclical features
    dataset['hr_sin'] = numpy.sin((dataset['TransactionDT_hour'] + dataset[
        'TransactionDT_min'] / 60.0) * (numpy.pi / 12.0))
    dataset['hr_cos'] = numpy.cos((dataset['TransactionDT_hour'] + dataset[
        'TransactionDT_min'] / 60.0) * (numpy.pi / 12.0))
    dataset['TransactionAmt_lg'] = numpy.log(dataset['TransactionAmt'])

    from sklearn.preprocessing import RobustScaler

    rob_scaler = RobustScaler()
    dataset['TransactionAmt_scaled'] = rob_scaler.fit_transform(
        dataset['TransactionAmt'].values.reshape(-1, 1))

    categorical_cols = ['DeviceType', 'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
    ohe = pandas.get_dummies(dataset[categorical_cols])
    ohe.drop('card6_debit or credit', axis=1, inplace=True)
    dataset = dataset.join(ohe)

    dataset.fillna(value={'addr1': -1.0}, inplace=True)
    dataset['addr1_scaled'] = rob_scaler.fit_transform(dataset['addr1'].values.reshape(-1, 1))

    train_cols = ["TransactionAmt_scaled", "hr_sin", "hr_cos", "addr1"] + list(ohe.columns)
    X, y = dataset[train_cols], dataset['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test



def logisticRegressionClassifier(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    print('logisticRegressionClassifier')

    clf = LogisticRegression(solver='lbfgs', max_iter=4000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)

    print(metrics.classification_report(y_test, preds))
    wandb.log({'accuracy_score': metrics.accuracy_score(y_test,preds)})

    wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, preds, pred_prob, clf.classes_,
                                  model_name='LogisticRegression', feature_names=None)


def xgbClassifier(X_train, X_test, y_train, y_test):
    print('xgboost')
    param = {}
    xg_train = xgboost.DMatrix(X_train, label=y_train)
    xg_test = xgboost.DMatrix(X_test, label=y_test)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['num_class'] = 2
    bst = xgboost.train(param, xg_train, 5, watchlist, callbacks=[wandb.xgboost.wandb_callback()])
    preds = bst.predict(xg_test)
    wandb.log({'accuracy_score': metrics.accuracy_score(y_test, preds)})
    # Use sklearn classifier API
    clf = xgboost.XGBClassifier(nthread=-1)
    clf.fit(X_train, y_train)
    print("done fitting")
    preds = clf.predict(X_test)

    pred_prob = clf.predict_proba(X_test)

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.termlog('Logged learning curve.')
    wandb.sklearn.plot_confusion_matrix(y_test, preds, clf.classes_)
    wandb.termlog('Logged confusion matrix.')
    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    wandb.termlog('Logged summary metrics.')
    wandb.sklearn.plot_class_proportions(y_train, y_test, clf.classes_)
    wandb.sklearn.plot_roc(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged roc curve.')
    wandb.sklearn.plot_precision_recall(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged precision recall curve.')


def randomForestClassifier(X_train, X_test, y_train, y_test):
    print('randomForestClassifier')

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    print(metrics.classification_report(y_test, preds))
    wandb.log({'accuracy_score': metrics.accuracy_score(y_test, preds)})

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.termlog('Logged learning curve.')
    wandb.sklearn.plot_confusion_matrix(y_test, preds, clf.classes_)
    wandb.termlog('Logged confusion matrix.')
    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    wandb.termlog('Logged summary metrics.')
    wandb.sklearn.plot_class_proportions(y_train, y_test, clf.classes_)
    wandb.termlog('Logged class proportions.')
    if (not isinstance(clf, naive_bayes.MultinomialNB)):
        wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, 'randomForestClassifier')
    wandb.termlog('Logged calibration curve.')
    wandb.sklearn.plot_roc(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged roc curve.')
    wandb.sklearn.plot_precision_recall(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged precision recall curve.')


def KNNClassifier(X_train, X_test, y_train, y_test):
    print('KNN')
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    print(metrics.classification_report(y_test, preds))

    wandb.log({'accuracy_score': metrics.accuracy_score(y_test, preds)})

    wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
    wandb.termlog('Logged learning curve.')
    wandb.sklearn.plot_confusion_matrix(y_test, preds, clf.classes_)
    wandb.termlog('Logged confusion matrix.')
    wandb.sklearn.plot_summary_metrics(clf, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    wandb.termlog('Logged summary metrics.')
    wandb.sklearn.plot_class_proportions(y_train, y_test, clf.classes_)
    wandb.termlog('Logged class proportions.')
    if (not isinstance(clf, naive_bayes.MultinomialNB)):
        wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, 'KNeighborsClassifier')
    wandb.termlog('Logged calibration curve.')
    wandb.sklearn.plot_roc(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged roc curve.')
    wandb.sklearn.plot_precision_recall(y_test, pred_prob, clf.classes_)
    wandb.termlog('Logged precision recall curve.')


def call_trainer(X_train, X_test, y_train, y_test):
    if wandb.config.model == 'xgboost':
        xgbClassifier(X_train, X_test, y_train, y_test)

    if wandb.config.model == 'logistic':
        logisticRegressionClassifier(X_train, X_test, y_train, y_test)

    if wandb.config.model == 'randomForest':
        randomForestClassifier(X_train, X_test, y_train, y_test)






class DataSweeper:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def preprocess_data(self):

        trainset = load_data()
        trainset = transform_time_columns(trainset)
        dataset = clean_data(trainset)
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessing(dataset)


    def train(self):
        wandb.init()
        if wandb.config.model == 'logistic':
            logisticRegressionClassifier(self.X_train, self.X_test, self.y_train, self.y_test)

        if wandb.config.model == 'randomForest':
            randomForestClassifier(self.X_train, self.X_test, self.y_train, self.y_test)
        if wandb.config.model == 'xgboost':
            xgbClassifier(self.X_train, self.X_test, self.y_train, self.y_test)

    def sweep(self):

        ##########
        # Sweeps #
        ##########
        sweep_config = {
            'method': 'random',  # grid, random
            'metric': {
                'name': 'accuracy_score',
                'goal': 'maximize'
            },
            'parameters': {

                'model': {
                    'values': ['randomForest', 'logistic', 'xgboost']
                }
            }
        }
        config_defaults = {

            'model': 'logistic'
        }
        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, function=self.train)


def main():

    data_sweeper = DataSweeper()
    data_sweeper.preprocess_data()
    data_sweeper.sweep()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
