import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import wandb
wandb.init(project="boston")

# Load data
def prepare_data():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Train model, get predictions
def regression_demo(X_train, X_test, y_train, y_test):
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Visualize model performance
    wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, 'Ridge')


def main():

    X_train, X_test, y_train, y_test = prepare_data()
    regression_demo(X_train, X_test, y_train, y_test)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
