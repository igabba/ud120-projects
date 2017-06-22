#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# your code goes here
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print accuracy_score(pred, y_test)
print "POI indentified:", sum(pred[x] for x in range(len(pred)) if pred[x] == 1)
print "Total number of people:", len(pred)

print "Precision:", precision_score(y_test, pred)
print classification_report(y_test, pred)
print confusion_matrix(y_test, pred)
