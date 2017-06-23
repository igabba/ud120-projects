#!/usr/bin/python

import pickle
import sys

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

sys.path.append("../tools/")

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
#  ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
# 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
# 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
features_list = ['poi', 'bonus', 'poi_related_messages', 'from_messages_poi_ratio']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# TODO buscar outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


# Task 3: Create new feature(s)
# TODO crear la relacion entre los from-to

def add_features(original_data_dict):
    """
    adds new features to de dictionary
    """
    for name in original_data_dict:
        try:
            if original_data_dict[name]["from_poi_to_this_person"] != 'NaN':
                original_data_dict[name]['from_messages_poi_ratio'] = 1. * original_data_dict[name]["from_poi_to_this_person"] / original_data_dict[name][
                    "from_messages"]
            else:
                original_data_dict[name]['from_messages_poi_ratio'] = 0

            if original_data_dict[name]["from_this_person_to_poi"] != 'NaN':
                original_data_dict[name]['person_to_poi_ratio'] = 1. * original_data_dict[name]["from_this_person_to_poi"] / original_data_dict[name][
                    "to_messages"]
            else:
                original_data_dict[name]['person_to_poi_ratio'] = 0

            poi_related_messages = original_data_dict[name]["from_poi_to_this_person"] + original_data_dict[name]["from_this_person_to_poi"] + \
                                   original_data_dict[name]["shared_receipt_with_poi"]
            if 'NaN' not in poi_related_messages:
                original_data_dict[name]['poi_related_messages'] = poi_related_messages
            else:
                original_data_dict[name]['poi_related_messages'] = 0
        except:
            original_data_dict[name]['poi_related_messages'] = 0

    return original_data_dict


def setup_clf_list():
    """
    Instantiates all classifiers of interstes to be used.
    """
    # List of tuples of a classifier and its parameters.
    clf_list = []

    #
    clf_naive = GaussianNB()
    params_naive = {}
    clf_list.append((clf_naive, params_naive))

    #
    clf_tree = DecisionTreeClassifier()
    params_tree = {"min_samples_split": [2, 5, 10, 20],
                   "criterion": ('gini', 'entropy')
                   }
    clf_list.append((clf_tree, params_tree))

    #
    clf_linearsvm = LinearSVC()
    params_linearsvm = {"C": [0.5, 1, 5, 10, 100, 10 ** 10],
                        "tol": [10 ** -1, 10 ** -10],
                        "class_weight": ['balanced']

                        }
    clf_list.append((clf_linearsvm, params_linearsvm))

    #
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = {"n_estimators": [20, 25, 30, 40, 50, 100]
                       }
    clf_list.append((clf_adaboost, params_adaboost))

    #
    clf_random_tree = RandomForestClassifier()
    params_random_tree = {"n_estimators": [2, 3, 5],
                          "criterion": ('gini', 'entropy')
                          }
    clf_list.append((clf_random_tree, params_random_tree))

    #
    clf_knn = KNeighborsClassifier()
    params_knn = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9], "p": [2, 3, 4]}
    clf_list.append((clf_knn, params_knn))

    #
    clf_log = LogisticRegression()
    params_log = {"C": [0.05, 0.5, 1, 10, 10 ** 2, 10 ** 5, 10 ** 10, 10 ** 20],
                  "tol": [10 ** -1, 10 ** -5, 10 ** -10],
                  "class_weight": ['balanced']
                  }
    clf_list.append((clf_log, params_log))

    #
    clf_lda = LinearDiscriminantAnalysis()
    params_lda = {"n_components": [0, 1, 2, 5, 10]}
    clf_list.append((clf_lda, params_lda))

    #
    logistic = LogisticRegression()
    rbm = BernoulliRBM()
    clf_rbm = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    params_rbm = {
        "logistic__tol": [10 ** -10, 10 ** -20],
        "logistic__C": [0.05, 0.5, 1, 10, 10 ** 2, 10 ** 5, 10 ** 10, 10 ** 20],
        "logistic__class_weight": ['balanced'],
        "rbm__n_components": [2, 3, 4]
    }
    clf_list.append((clf_rbm, params_rbm))

    return clf_list


def search_best_clf():
    # type: () -> GridSearchCV
    """
    Search for the best classifier and return it
    :rtype: GridSearchCV
    """
    f1score = 0
    best_clf = None
    for clftemp, params in setup_clf_list():
        scorer = make_scorer(f1_score)
        clftemp = GridSearchCV(clftemp, params, scoring=scorer)
        clftemp = clftemp.fit(features_train, labels_train)
        pred = clftemp.predict(features_test)
        f1_score_temp = f1_score(labels_test, pred)
        if f1_score_temp > f1score:
            f1score = f1_score_temp
            best_clf = clftemp.best_estimator_

    print "best F1 Score: ", f1score
    print "best Classifier: ", best_clf
    pred = best_clf.predict(features_test)
    print classification_report(labels_test, pred)
    print confusion_matrix(labels_test, pred)

    return best_clf


data_dict = add_features(data_dict)
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info: 
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
from tester import test_classifier

clf = search_best_clf()
dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)
