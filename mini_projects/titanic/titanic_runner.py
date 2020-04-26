import time

import pandas

from etl import data_loader
from etl import data_splitter

FOLD_MAPPINGS = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}

def load_data():
    return data_loader.load_titanic()


def shuffle_data(data_frame: pandas.DataFrame):
    return data_frame.sample(frac=1.0).reset_index(drop=True)


def preproces_data():
    fold = 0
    target_field = 'Survived'
    data_frame = load_data()
    data_frame = shuffle_data(data_frame)

    data_frame = data_splitter.create_train_validation_sets(data_frame, 'Survived')

    train_data_frame = data_frame[data_frame['kfold'].isin(FOLD_MAPPINGS.get(fold))]
    valid_data_frame = data_frame[data_frame['kfold'] == fold]

    train_data_frame = train_data_frame.drop(columns=[target_field, 'kfold'])
    valid_data_frame = valid_data_frame.drop(columns=[target_field, 'kfold'])


def main():
    preproces_data()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
