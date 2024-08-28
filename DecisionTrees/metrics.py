import pandas as pd
from typing import Union
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size
    # TODO: Write here
    return (y_hat==y).sum()/len(y)

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert cls in y and cls in y_hat
    return ((y_hat == cls) & (y == cls)).sum() / (y_hat == cls).sum()


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert cls in y and cls in y_hat
    return ((y_hat == cls) & (y == cls)).sum() / (y == cls).sum()

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size !=0
    return np.sqrt(((y_hat-y)**2).sum()//len(y))

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size !=0
    return ((y_hat-y).abs()).sum()/len(y)

