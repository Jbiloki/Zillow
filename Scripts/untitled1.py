# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:23:47 2017

@author: Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    print(train2016.shape)