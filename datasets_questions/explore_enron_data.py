#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data = pd.read_pickle("../final_project/final_project_dataset.pkl")
enron_data = pd.DataFrame(enron_data)
enron_data =  enron_data.T

print enron_data['poi']


