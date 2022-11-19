# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/yunseokddi/OpenSource_assignment_2

import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_path):
    dataset_df = pd.read_csv(dataset_path)

    return dataset_df


def dataset_stat(dataset_df):
    n_feats = len(dataset_df)
    n_class0 = len(dataset_df[dataset_df['target'] == 0])
    n_class1 = len(dataset_df[dataset_df['target'] == 1])

    return n_feats, n_class0, n_class1


def split_dataset(dataset_df, testset_size):
    x_train, x_test = train_test_split(dataset_df, test_size=testset_size)

    y_train = x_train['target']
    y_test = x_test['target']

    del x_train['target']
    del x_test['target']

    return x_train, x_test, y_train, y_test


def decision_tree_train_test(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)

    return acc, prec, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)

    return acc, prec, recall


def svm_train_test(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    clf = svm.SVC()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred)
    recall = recall_score(y_true=y_test, y_pred=y_pred)

    return acc, prec, recall


def print_performances(acc, prec, recall):
    # Do not modify this function!
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)


if __name__ == '__main__':
    # Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features: ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)

    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)
