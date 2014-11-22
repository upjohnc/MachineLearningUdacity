#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):

    import numpy as np
    import pandas as pd

    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    temp = np.absolute(predictions - net_worths)
    temp = np.sort(temp, axis = 0)
    temp = pd.DataFrame(temp)
    limit = int(temp.shape[0]*0.9)
    temp = np.array(temp.ix[0:limit])
    for i in range(limit):
        cleaned_data.append([ages[i], net_worths[i], temp[i]])
    # cleaned_data = \
    # print


    
    return cleaned_data

