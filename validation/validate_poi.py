#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]
print len(data_dict)
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


print len(features), len(labels)
### it's all yours from here forward!  
from sklearn import cross_validation

featTrain, featTest, labelTrain, labelTest = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)

from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(featTrain, labelTrain)

predict = clf.predict(featTest)

print accuracy_score(labelTest, predict)
