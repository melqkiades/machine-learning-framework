import os
import pandas
import joblib
import numpy
import time

from mini_projects.cat_in_the_dat.citd_constants import MODELS_FOLDER, TEST_DATA


def predict(test_data_path, model_type, model_path):
    data_frame = pandas.read_csv(test_data_path)
    test_idx = data_frame["id"].values
    predictions = None

    for FOLD in range(5):
        data_frame = pandas.read_csv(test_data_path)
        encoders = joblib.load(
            os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(
            os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        for column in encoders:
            label_encoder = encoders[column]
            data_frame.loc[:, column] = data_frame.loc[:, column].astype(str).fillna("NONE")
            data_frame.loc[:, column] = label_encoder.transform(data_frame[column].values.tolist())

        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))

        data_frame = data_frame[cols]
        preds = clf.predict_proba(data_frame)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pandas.DataFrame(numpy.column_stack((test_idx, predictions)),
                       columns=["id", "target"])
    return sub


def main():
    submission = predict(test_data_path=TEST_DATA,
                         model_type="randomforest",
                         model_path=MODELS_FOLDER)
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"{MODELS_FOLDER}/rf_submission.csv", index=False)

    # The submission with random forests and 200 trees score was 0.74753
    # On the private leaderboard 0.74498, on the public leaderboard 0.75143

    # Train score after 5-fold average is 0.740586454
    # The submission with random forests and 200 trees score was 0.74718
    # On the private leaderboard 0.74498, on the public leaderboard 0.75143
    # Training time 10 miinutes
    # Prediction time 3 minutes


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
