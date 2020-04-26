import glob
import numpy # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle, islice
from sklearn.neighbors import BallTree, KDTree, DistanceMetric
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import wandb


TRAIN_FILE = '/Users/fpena/Datasets/titanic/train_wandb.csv'

# Load data
def load_data():
    return pandas.read_csv(TRAIN_FILE)

# Get numeric labels for each of the string labels, to make them compatible with our model
def get_class_ids(labels):
    labels_to_class = {'Did not Survive': 0, 'Survived': 1}
    return numpy.array([labels_to_class[alabel] for alabel in labels])


def get_named_labels(labels, numeric_labels):
    return numpy.array([labels[num_label] for num_label in numeric_labels])


def prepare_classifier_data(data_frame):
    # Remove target variables label (and class)
    features = list(set(data_frame.columns) - {'Survived', 'Name'})
    X = data_frame[features]
    y = data_frame['Survived']
    labels = ['Did not Survive', 'Survived']
    X = X[:50000]
    X = X.replace("", numpy.nan, regex = True)
    y = y[:50000]

    scaler = MinMaxScaler(feature_range=(0,1))
    features_scaled = scaler.fit_transform(X)
    # Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, labels, features

# Clustering - predict particle clusters without labels
def clustering_demo(X_train, labels):
    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=1)
    cluster_labels = kmeans.fit_predict(X_train)
    wandb.init(project="titanic", name='KMeans', reinit=True)
    label_names = get_named_labels(labels, cluster_labels)
    wandb.sklearn.plot_clusterer(kmeans, X_train, cluster_labels, labels, 'KMeans')
    # wandb.sklearn.plot_elbow_curve(model, X_train)


# Classification - predict pulsar
def classification_demo(X_train, X_test, y_train, y_test, labels, features):
    # Train a model, get predictions
    log = linear_model.LogisticRegression(random_state=4)
    knn = KNeighborsClassifier(n_neighbors=2)
    dtree = DecisionTreeClassifier(random_state=4)
    rtree = RandomForestClassifier(n_estimators=100, random_state=4)
    svm = SVC(random_state=4, probability=True)
    nb = GaussianNB()
    gbc = GradientBoostingClassifier()
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.01, random_state=42,
                                 base_estimator=DecisionTreeClassifier(max_depth=8,
                                 min_samples_leaf=10, random_state=42))

    model_algorithm(log, X_train, y_train, X_test, y_test, 'LogisticRegression', labels, features)
    model_algorithm(knn, X_train, y_train, X_test, y_test, 'KNearestNeighbor', labels, features)
    model_algorithm(dtree, X_train, y_train, X_test, y_test, 'DecisionTree', labels, features)
    model_algorithm(rtree, X_train, y_train, X_test, y_test, 'RandomForest', labels, features)
    model_algorithm(svm, X_train, y_train, X_test, y_test, 'SVM', labels, features)
    model_algorithm(nb, X_train, y_train, X_test, y_test, 'NaiveBayes', labels, features)
    model_algorithm(adaboost, X_train, y_train, X_test, y_test, 'AdaBoost', labels, features)
    model_algorithm(gbc, X_train, y_train, X_test, y_test, 'GradientBoosting', labels, features)


def model_algorithm(clf, X_train, y_train, X_test, y_test, name, labels, features):
    clf.fit(X_train, y_train)
    y_probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    wandb.init(project="titanic", name=name, reinit=True)
    # wandb.sklearn.plot_roc(y_test, y_probas, labels, reinit = True)
    wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train,
                    y_test, y_pred, y_probas, labels, True, name, features)


def prepare_regression_data(data_frame):
    features = list(set(data_frame.columns) - {'Age'})
    X = data_frame[features]
    y = data_frame['Age']
    X = X[:10000]
    y = y[:10000]

    # Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                        train_size=0.90,
                                                        test_size=0.10)

    return X_train, X_test, y_train, y_test


# Regression - TrackP - particle momentum
def regression_demo(X_train, X_test, y_train, y_test):

    # Train a model, get predictions
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Visualize model performance
    wandb.init(project="titanic", name='Ridge', reinit=True)
    wandb.sklearn.plot_regressor(reg, X_train, X_test,
                                  y_train, y_test, 'Ridge')


def main():

    data_frame = load_data()
    X_train, X_test, y_train, y_test, labels, features = prepare_classifier_data(data_frame)
    clustering_demo(X_train, labels)
    classification_demo(X_train, X_test, y_train, y_test, labels, features)

    X_train, X_test, y_train, y_test = prepare_regression_data(data_frame)
    regression_demo(X_train, X_test, y_train, y_test)


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
