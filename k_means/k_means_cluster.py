#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy as np
import sys

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

from feature_format import featureFormat, targetFeatureSplit

sys.path.append("../tools/")


def draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

# the input features we want to use 
# can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)
scaled_array = []
salary_array = []
stock_array = []
for k, v in data_dict.items():
    salary = 0.
    exercised_stock_options = 0.
    if v['salary'] != 'NaN':
        salary = v['salary']
    if v['exercised_stock_options'] != 'NaN':
        exercised_stock_options = v['exercised_stock_options']

    salary_array.append(salary)
    stock_array.append(exercised_stock_options)

salary_array.append(200000.)
min_max_scaler = preprocessing.MinMaxScaler()
sa_minmax = min_max_scaler.fit_transform(salary_array)
print sa_minmax[len(sa_minmax) - 1]
stock_array.append(1000000.)
st_minmax = min_max_scaler.fit_transform(stock_array)
print st_minmax[len(st_minmax) - 1]

finance_features2 = zip(sa_minmax, st_minmax)
# in the "clustering with 3 features" part of the mini-project,
# you'll want to change this line to 
# for f1, f2, _ in finance_features:
# (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features2:
    plt.scatter(f1, f2)
plt.show()

# cluster here; create predictions of the cluster labels
# for the data and store them to a list called pred
kmeans = KMeans(n_clusters=2)

pred = kmeans.fit_predict(finance_features2)

# rename the "name" parameter when you change the number of features
# so that the figure gets saved to a different file
try:
    draw(pred, finance_features2, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

stock = []
for i in data_dict:
    if data_dict[i]["exercised_stock_options"] == 'NaN':
        pass
    else:
        stock.append(float(data_dict[i]["exercised_stock_options"]))
ma = max(stock)
mi = min(stock)
print "Exercised stock options maximum: ", ma, " minimum: ", mi
print float(1000000 - mi) / (ma - mi)

salary = []
for i in data_dict:
    if data_dict[i]["salary"] == 'NaN':
        pass
    else:
        salary.append(float(data_dict[i]["salary"]))
maxSalary = max(salary)
minSalary = min(salary)
print "Salary maximum: ", maxSalary, " minimum: ", minSalary
print float(1000000 - minSalary) / (maxSalary - minSalary)
