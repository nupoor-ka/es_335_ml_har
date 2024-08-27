from typing import Union
import pandas as pd
import numpy as np

'''
Variations of the problem:
1) Discrete I/P, Discrete O/P => classficiation 
2) Discrete I/P, Real O/P => regression 
3) Real I/P, Discrete O/P => classification
3) Real I/P, Real O/P => regression
'''

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy. Defined only for 1) and 3). 
    """
    # Assert that y_hat and y have the same length
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."

    
    numerator = (y_hat == y).sum()
    #print(f"Numerator (correct predictions): {numerator}")

   
    denominator = y.size
    #print(f"Denominator (total predictions): {denominator}")

    
    accuracy = numerator / denominator
    return float(accuracy) 


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision. Defined only for 1) and 3). 
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    #y_hat = y = class
    
    # Calculate predicted positives
    predicted_positives = (y_hat == cls).sum()
    #y_hat = class
    
    return true_positives / predicted_positives if predicted_positives != 0 else 0.0

    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision. Defined only for 1) and 3). 
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."

    true_positives = ((y_hat == cls) & (y == cls)).sum()
    
    # Calculate predicted positives
    gnd_positives = (y == cls).sum()
    
    return true_positives / gnd_positives if gnd_positives != 0 else 0.0




def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse). Defined only for 2) and 4).
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y.size!=0, "Ground Truth array is 0"
    assert y_hat.size!=0, "Predicition array is 0"

    y_c = y.copy()
    y_hat_c = y_hat.copy()

    y_hat_c = np.array(y_hat_c)
    y_c = np.array(y_c)
    numerator = np.sum((y_hat_c-y_c)**2)
    denominator = y_c.size

    return np.sqrt(numerator/denominator) 





def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae). Defined only for 2) and 4).
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y.size, "Ground Truth array is 0"
    assert y_hat.size, "Predicition array is 0"

    y_c = y.copy()
    y_hat_c = y_hat.copy()

    y_hat_c = np.array(y_hat_c)
    y_c = np.array(y_c)
    numerator = np.sum((np.abs(y_hat_c-y_c)))

    denominator = y_c.size

    return numerator/denominator
