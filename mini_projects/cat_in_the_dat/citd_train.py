
import pandas
import time
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib


# 0.75091
from mini_projects.cat_in_the_dat.citd_constants import MODELS_FOLDER
from mini_projects.cat_in_the_dat.citd_constants import TRAINING_DATA_FOLDS, \
    TEST_DATA

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}


MODEL = "randomforest"


FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}


def train(fold):
    data_frame = pandas.read_csv(TRAINING_DATA_FOLDS)
    test_df = pandas.read_csv(TEST_DATA)

    train_df, valid_df = create_train_validation_sets(data_frame, fold)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    # Making sure that the order of the variables is the same in train and valid
    # This is not needed most of the time
    valid_df = valid_df[train_df.columns]

    create_label_encoders(train_df, valid_df, test_df, fold)

    # data is ready to train
    clf = MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(clf, f"{MODELS_FOLDER}/{MODEL}_{fold}.pkl")


def create_train_validation_sets(data_frame: pandas.DataFrame, fold: int):
    train_df = data_frame[
        data_frame.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
    valid_df = data_frame[data_frame.kfold == fold].reset_index(drop=True)

    return train_df, valid_df


def create_label_encoders(train_df, valid_df, test_df, fold):
    # TODO: See if you really need a label encoder per fold

    label_encoders = {}
    for column in train_df.columns:
        label_encoder = preprocessing.LabelEncoder()
        train_df.loc[:, column] = train_df.loc[:, column].astype(str).fillna(
            "NONE")
        valid_df.loc[:, column] = valid_df.loc[:, column].astype(str).fillna(
            "NONE")
        test_df.loc[:, column] = test_df.loc[:, column].astype(str).fillna(
            "NONE")
        label_encoder.fit(
            train_df[column].values.tolist() +
            valid_df[column].values.tolist() +
            test_df[column].values.tolist())
        train_df.loc[:, column] = label_encoder.transform(
            train_df[column].values.tolist())
        valid_df.loc[:, column] = label_encoder.transform(
            valid_df[column].values.tolist())
        label_encoders[column] = label_encoder

    joblib.dump(label_encoders, f"{MODELS_FOLDER}/{MODEL}_{fold}_label_encoder.pkl")
    joblib.dump(train_df.columns, f"{MODELS_FOLDER}/{MODEL}_{fold}_columns.pkl")



def main():
    for fold in range(5):
        train(fold)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
