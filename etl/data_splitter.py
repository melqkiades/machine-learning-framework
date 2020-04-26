import pandas
from sklearn import model_selection


def create_train_validation_sets(
        data_frame: pandas.DataFrame, target_col, num_folds=5) -> pandas.DataFrame:
    kf = model_selection.StratifiedKFold(n_splits=num_folds,
                                         shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=data_frame, y=data_frame[target_col].values)):
        data_frame.loc[val_idx, 'kfold'] = fold

    return data_frame
