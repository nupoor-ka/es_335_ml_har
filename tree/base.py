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
        self.tree = {}
        self.current_depth = 0
        self.type = ""

    def fit(self, X: pd.DataFrame, y: pd.Series, branch_label = '1_') -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        #self notes 
        #the df I recieve, regardless of column, if real input, all columns real, if discrete output, all columns discrete 
        
        print(self.current_depth) #####

        if self.current_depth == self.max_depth:
            return None

        if self.current_depth==0:
            self.type = check_type(X, y)
            print(self.type)  #######

            if self.type[0]=="d":
                X = one_hot_encoding(X, columns = X.columns)

                
        if self.type[1] == 'd':
            if len(y.unique())==1:
                self.tree[branch_label] = y[0]
                return None
        if y.size==1:
            self.tree[branch_label] = y[0]
            return None
        
            

        if self.type[0] == "r":
            #real input 

            fnn = opt_split_attribute_real_input #function for real input split 

            if self.type[1]=="r":
                #Real input, Real output
                df_feature_importance_and_split_point = fnn(X, y, X.columns, self.type) #pd.DataFrame["attribute","split_point","sel_or_infogain"])
                (X_left, y_left), (X_right, y_right) =split_data(X, y, df_feature_importance_and_split_point["attribute"][0], df_feature_importance_and_split_point["split_point"][0])
            else:
                #Real input, Discrete output 
                df_feature_importance_and_split_point = fnn(X, y, X.columns, self.type) #pd.DataFrame["attribute","split_point","sel_or_infogain"])
                (X_left, y_left), (X_right, y_right) =split_data(X, y, df_feature_importance_and_split_point["attribute"][0], df_feature_importance_and_split_point["split_point"][0])
            
            self.current_depth+=1
            print(self.tree) ##########
            self.tree[branch_label] = {'attribute':df_feature_importance_and_split_point["attribute"][0], 'split_value':df_feature_importance_and_split_point["split_point"][0], 'right_label':str(branch_label+'1_'), 'left_label':str(branch_label+'2_')}
            self.fit(self, X_left, y_left, branch_label = str(branch_label+'2_'))
            self.fit(self, X_right, y_right, branch_label = str(branch_label+'1_'))

        else:
            #discrete input 
            fnn = opt_split_attribute_discrete_input #function for discrete input split 
            #def opt_split_attribute_discrete_input(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):

            if self.type[1]=="r":
                #discrte ip, real op 
                features_importance = fnn(X,y,self.criterion,X.columns, self.type)
            else:
                #discrete ip, discrete op 

                features_importance = fnn(X,y,self.criterion,X.columns,self.type)
                
                #split_data(X: pd.DataFrame, y: pd.Series, attribute, value) # return (X_left, y_left), (X_right, y_right)
                




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
