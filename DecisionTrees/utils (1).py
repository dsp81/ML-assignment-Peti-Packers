from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import xlogy


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
     encoded_df = pd.get_dummies(X, drop_first=True)
     return encoded_df


def check_ifreal(y: pd.Series,real_ratio=0.2) -> bool: #change
    # Assuming that numeric data types (float or int) are more likely to represent continuous values
    if pd.api.types.is_numeric_dtype(y)==0:
        return False
    elif y.dtype == 'float64':
        return True
    else:
        if y.nunique()/len(y)< real_ratio:
            return False
        else:
            return True

def real_or_not(X:pd.DataFrame,y:pd.Series):

          a={}
          for feature in X.columns:
            a[feature]=check_ifreal(X[feature])
          a['label']=check_ifreal(y)
          return a
      
def gini_index(Y: pd.Series) -> float:
    P=Y.value_counts()/len(Y)
    return 1-sum(P**2)

def gini_discrete(Y: pd.Series, attr: pd.Series) -> float:
    S=attr.value_counts()
    p=S/len(attr)
    weighted_gini=0
    for label in attr.unique():
        ind=attr.index[attr==label]
        weighted_gini += p[label] * gini_index(Y[ind])
    return weighted_gini

def gini_real(Y: pd.Series, ind:int) -> float:
    return (gini_index(Y[:ind])) * len(Y[:ind]) / len(Y) + (gini_index(Y[ind+1:])) * len(Y[ind+1:]) / len(Y)

def entropy(Y: pd.Series) -> float:
    P=Y.value_counts()/len(Y)
    result = xlogy(P,P)/np.log(2)
    return -sum(result)

def weighted_mse(Y_subset: pd.Series) -> float:
    mean_subset = Y_subset.mean()
    return ((Y_subset - mean_subset) ** 2).mean()

def information_gainR(Y: pd.Series, ind: int,is_real)-> float: #for real input 
    if(is_real['label']):
        initial_mse = ((Y - Y.mean()) ** 2).mean()
        weighted_total_mse = weighted_mse(Y[:ind])* len(Y[:ind]) / len(Y) + weighted_mse(Y[ind+1:])* len(Y[ind+1:]) / len(Y)
        return initial_mse- weighted_total_mse
    else: #discrete out
        weighted_entropy= (entropy(Y[:ind])) * len(Y[:ind]) / len(Y) + (entropy(Y[ind+1:])) * len(Y[ind+1:]) / len(Y)
        return entropy(Y)-weighted_entropy


def information_gainD(y: pd.Series, attr: pd.Series,is_real) -> float: #discrete input
    if(is_real['label']): #discrete_real
      initial_mse = ((y - y.mean()) ** 2).mean()
      weighted_total_mse = 0.0
      for category in attr.unique():
          subset = attr == category
          N_subset = len(y[subset])
          mse_subset = weighted_mse(y[subset])
      weighted_mse_value = (N_subset / len(y)) * mse_subset
      weighted_total_mse += weighted_mse_value
      return initial_mse- weighted_total_mse
    else: # disc_disc
      S=attr.value_counts()
      p=S/len(attr)
      weighted_entropy=0
      for label in attr.unique():
          ind=attr.index[attr==label]
          weighted_entropy += p[label] * entropy(y[ind])
      return entropy(y)-weighted_entropy

def split_real_column(feature, Y: pd.Series,criterion,is_real): 
    #returns value at which split to be made and value of gini/infogain
        df = pd.DataFrame({'feature': feature, 'label': Y})
        df_sorted = df.sort_values('feature')
    
        if criterion == "gini_index":
            ginimin = float('inf')
            ind=0
            for i in range(len(Y)-1 ):
                if df_sorted['label'].iloc[i] != df_sorted['label'].iloc[i + 1]:
                    gini = gini_real(df_sorted['label'], i)
                    if gini <= ginimin:
                        ind = i
                        ginimin = gini
            mean = (df_sorted['feature'].iloc[ind] + df_sorted['feature'].iloc[ind + 1]) / 2
            return mean, ginimin

        elif criterion == "information_gain":
            max_inf=0
            ind=0
            for i in range(len(Y)-1):
                    inf = information_gainR(df_sorted['label'], i,is_real)
                    if max_inf <= inf:
                        ind = i
                        max_inf = inf
            mean = (df_sorted['feature'].iloc[ind] + df_sorted['feature'].iloc[ind + 1]) / 2
            return mean,max_inf

def opt_split_attributeR(X: pd.DataFrame, y: pd.Series,criterion, features: pd.Series,is_real): #real_input
  
    if criterion == "gini_index":
        ginimin = float('inf')
        for feature in features:
            mean,gini=split_real_column(X[feature],y,criterion,is_real)
            if gini <= ginimin:
                    meanmin = mean
                    ginimin = gini
                    featuremin=feature
        return featuremin,meanmin
    elif criterion =="information_gain":
        maxinf=0
        for feature in features:
            mean,inf=split_real_column(X[feature],y,criterion,is_real)
            if maxinf <= inf:
                    meanmax = mean
                    maxinf = inf
                    featuremax=feature
        return featuremax, meanmax

def opt_split_attributeD(X: pd.DataFrame, y: pd.Series,criterion, features: pd.Series,is_real): #discrete input

     if criterion == "information_gain":
            best_feature, max_inf_gain = max(((attr, information_gainD(y, X[attr],is_real)) for attr in features),
                                             key=lambda x: x[1])
     elif criterion == "gini_index":
            best_feature, min_gini = min(((attr, gini_discrete(y, X[attr])) for attr in features),
                                          key=lambda x: x[1])
     else:
            raise ValueError(f"Unsupported criterion: {criterion}")
     return best_feature

def split_dataR(X: pd.DataFrame, y: pd.Series, attribute, mean):

    mask_less_than_mean = X[attribute] < mean
    X_left = X[mask_less_than_mean]
    X_right = X[~mask_less_than_mean]
    y_left = y[mask_less_than_mean]
    y_right = y[~mask_less_than_mean]
    if(len(y_left)==0):
        y_left=None
        X_left=None
    if(len(y_right)==0):
        y_right=None
        X_right=None

    return X_left, X_right, y_left, y_right


def split_dataD(X: pd.DataFrame, y: pd.Series, attribute, value):
    new_y=y[X[attribute]==value]
    new_x=X.mask(X[attribute]!=value).dropna().drop(attribute,axis=1)
    return new_x,new_y