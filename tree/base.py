"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)



@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        #self notes 
        #the df I recieve, regardless of column, if real input, all columns real, if discrete output, all columns discrete 
        
        if self.max_depth == 0:
            return

        

        if check_ifreal(X[0])==True:
            #real input 

            fnn = opt_split_attribute_real_input #function for real input split 

            if check_ifreal(y)==True:
                #Real input, Real output
                df_feature_importance_and_split_point = fnn(X, y, X.columns) 
            else:
                #Real input, Discrete output 
                df_feature_importance_and_split_point = fnn(X, y, X.columns) 
        else:
            #discrete input 
            fnn = opt_split_attribute_discrete_input #function for discrete input split 
            #def opt_split_attribute_discrete_input(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):

            if check_ifreal(y)==True:
                #discrte ip, real op 
                features_importance = fnn(X,y,self.criterion,X.columns)
            else:
                #discrete ip, discrete op 
                self.max_depth -= 1
                features_importance = fnn(X,y,self.criterion,X.columns)
                X = X.drop(features_importance, axis=1)
                self.tree.append([features_importance])

                




    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        print(self.tree)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
