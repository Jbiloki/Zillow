# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:23:47 2017

@author: Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

class DataAnalysis:
    def __init__(self):
        self = self


if __name__ == "__main__":
    da = DataAnalysis()
    selectedCols =  ['parcelid','airconditioningtypeid', 'bathroomcnt', 'bedroomcnt','buildingclasstypeid','buildingqualitytypeid','calculatedfinishedsquarefeet','lotsizesquarefeet']
    trainCols = ['parcelid','logerror']
    frame2016 = pd.read_csv('../Data/properties_2016.csv', usecols = selectedCols).fillna(0)
    train2016 = pd.read_csv('../Data/train_2016.csv', usecols = trainCols)
    frame2016.set_index(['parcelid'], inplace=True)
    train2016.set_index(['parcelid'], inplace=True)
    merge = pd.merge(frame2016,train2016,right_index=True,left_index=True)

    #parcelIds = train2016['parcelid'].values
    #frame2016 = frame2016[frame2016.index.isin(parcelIds)]
    #frameIds = frame2016.index.values
    #train2016 = train2016[train2016['parcelid'].isin(frameIds)]

    
    reg = linear_model.Ridge (alpha = .5)
    reg.fit(merge.iloc[:,1:7].astype(float),merge['logerror'].astype(float))
    guesses = []
    for index,row in merge.iterrows():
        #print(row[7])
        guesses.append(np.asarray(row[1:7]))
        #guesses.append(reg.predict(predict.reshape(1,-1)))

    #print(guesses)
    Y_True = merge['logerror'].values.reshape(-1,1)
    Y_Pred = np.array(guesses)
    predicted = cross_val_predict(reg, guesses[:5000], Y_True[:5000], cv=10)
    true = Y_True[:5000]
    plt.scatter(true,predicted)
    print("Mean squared error: %.2f" % np.mean((reg.predict(guesses[:1000]) - Y_True[:1000]) ** 2))
    plt.show()