"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame, columns = None) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    #we need to specify which columns need to be one-hot coded

    return pd.get_dummies(X, columns=columns, dtype=int)



def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    #inuitive approach, underoot of length < unique => real

    return np.sqrt(y.size) <= y.unique().size 


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    classes = Y.unique()
    counts_dict = Y.value_counts().to_dict()
    length = Y.size

    p = np.zeros(len(classes))
    entropy=0.0

    for i in range(len(classes)):
        p[i]=counts_dict[classes[i]]/length
        entropy+=-p[i]*np.log2(p[i])

    return entropy        


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    classes = Y.unique()
    counts_dict = Y.value_counts().to_dict()
    length = Y.size

    p = np.zeros(len(classes))
    sum=0.0

    for i in range(len(classes)):
        p[i]=counts_dict[classes[i]]/length
        sum+=p[i]**2

    return 1-sum

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    mean = np.mean(Y)
    
    return np.sum((mean - Y)**2)/Y.size 

def squared_error(Y: pd.Series)-> float:
    mean_y = np.mean(Y)
    return np.sum((mean_y -Y)**2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str):
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    
    Text: "gini_index", "entropy", "mse", "squared_error"

    return: float for discrete input, dataframe for real input
    """
    #FIX FOR REAL INPUT OR attr.dtype = float

    if check_ifreal(attr)==False:
        #discrete input    => both cases 
        if criterion == "info_gain":
            fn = entropy
        
        if criterion == "gini_index":
            fn = gini_index
        
        if criterion =="mse":
            fn = mse

        info_gain = fn(Y)
        reduction=0.0
        df = pd.DataFrame({"value":Y,"attribute":attr})

        for att_val in attr.unique():
            df_filtered = df[df["attribute"]==att_val]
            weight = len(df_filtered)/len(df)
            reduction+= weight*fn(df_filtered["value"])
        
        info_gain -= reduction

        return info_gain
    else:
        #real input
        #step 1: I need to sort my table based on attributes,
        sorted_indices = attr.sort_values().index

        attr_sorted = attr.loc[sorted_indices]
        Y_sorted = Y.loc[sorted_indices]

        if criterion == "info_gain":
            fn = entropy
        
        if criterion == "gini_index":
            fn = gini_index
        
        if criterion =="squared_error":
            fn = squared_error
        
        #Real input, Discrete output (Entropy and Gini are relevant here)

        if check_ifreal(Y)==False:
            #potential split number is one less than number of rows 
            info_gain = np.array([fn(Y_sorted)]*(Y.size-1))

            for i in range(info_gain.shape[0]):
                lhs_entropy = fn(Y_sorted[:i+1]) #could be gini as well
                rhs_entropy = fn(Y_sorted[i+1:])

                weighted_entropy = (Y_sorted[:i+1].size/Y_sorted.size)*lhs_entropy + (Y_sorted[i+1:].size/Y_sorted.size)*rhs_entropy

                info_gain[i] -= weighted_entropy

            attr_sorted=(attr_sorted[:-1] + attr_sorted[1:]) / 2 #(taking midpoints for split)

            return pd.DataFrame({"Split values":attr_sorted,"Info Gain":info_gain})
        
        #Real input, Real output 
        else:
            loss_across = np.zeros(Y.size-1)

            for i in range(info_gain.shape[0]):
                lhs_se = fn(Y_sorted[:i+1]) #squared error is the function used 
                rhs_se = fn(Y_sorted[i+1:])

                loss_across[i] = lhs_se + rhs_se #we will pick the min loss to split 

            return pd.DataFrame({"Split Values":attr_sorted,"loss":loss_across})



def opt_split_attribute_discrete_input(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    all: if false, by default, returns attribute to make split on, if true, gives a df with info gain of all attributes

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).


    number_of_attributes = features.size 

    info_gain_arr = np.zeros(number_of_attributes)

    for i in range(number_of_attributes):
        info_gain_arr[i]=information_gain(y, X[features[i]], criterion)

    index_max = np.argmax(info_gain_arr)

    return features[index_max]
    

def opt_split_attribute_real_input(X: pd.DataFrame, y: pd.Series, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    This function works for all the cases with real input.
    
    features: pd.Series is a list of all the attributes we have to split upon (assuming all features are real)

    return: attribute to split upon
    """

    number_of_attributes = features.size 

    if check_ifreal(y)==False:
        df_final = pd.DataFrame(columns=["attribute","split_point","sel_or_infogain"])

        for i in range(number_of_attributes):
            feature = features[i]
            df = pd.DataFrame({"Label": y, "Attribute":X[feature]})

            df_returned =information_gain(df["Label"],df["Attribute"], "info_gain") #what i recieved pd.DataFrame({"Features":features,"Information Gain":info_gain_arr}).sort_values(by="Information Gain")

            df_returned = df_returned.sort_values(by="Info Gain", ascending=False) 
            df_returned.reset_index(drop=True, inplace=True)

            df_final.iloc[i,0]=feature
            df_final.iloc[i,1]=df_returned["Split values"]
            df_final.iloc[i,2]=df_returned["loss"]
        
        return df_final
    else:
        #Real input, real output
        df_final = pd.DataFrame(columns=["attribute","split_point","sel_or_infogain"])

        for i in range(number_of_attributes):
            feature = features[i]
            df = pd.DataFrame({"Label": y, "Attribute":X[feature]})

            df_returned =information_gain(df["Label"],df["Attribute"], "squared_error") #what i recieved pd.DataFrame({"Split Values":attr_sorted,"loss":loss_across})


            df_returned = df_returned.sort_values(by="loss", ascending=True) #the most minimum loss is what is best
            df_returned.reset_index(drop=True, inplace=True)

            df_final.iloc[i,0]=feature
            df_final.iloc[i,1]=df_returned["Split values"]
            df_final.iloc[i,2]=df_returned["loss"]
        
        return df_final



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.


    mask = X[attribute] <= value if pd.api.types.is_numeric_dtype(X[attribute]) else X[attribute] == value

    X_left, y_left = X[mask], y[mask]
    X_right, y_right = X[~mask], y[~mask]

    return (X_left, y_left), (X_right, y_right)