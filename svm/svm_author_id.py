#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C = 10000.)
#10.0 -
# predicting time: 0.195 s
# predicting time: 2.351 s
# 0.616040955631

#100.-
# predicting time: 0.544 s
# predicting time: 2.355 s
# 0.616040955631

# 1000. -
# predicting time: 0.223 s
# predicting time: 2.312 s
# 0.821387940842

# 10000 -
# training time: 0.178 s
# predicting time: 1.749 s
# 0.892491467577
#
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

temp = pred == 1

print temp.sum()

print accuracy_score(labels_test, pred)

#########################################################


