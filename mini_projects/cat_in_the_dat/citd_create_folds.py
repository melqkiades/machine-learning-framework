import pandas
import time
from sklearn import model_selection

from mini_projects.cat_in_the_dat.citd_constants import TRAINING_DATA_FOLDS, \
    TRAINING_DATA


def create_folds():
    data_frame = pandas.read_csv(TRAINING_DATA)
    data_frame["kfold"] = -1

    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False,
                                         random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=data_frame, y=data_frame.target.values)):
        print(len(train_idx), len(val_idx))
        data_frame.loc[val_idx, 'kfold'] = fold

    data_frame.to_csv(TRAINING_DATA_FOLDS, index=False)


def main():
    create_folds()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
