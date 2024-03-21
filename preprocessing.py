# Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np



def input_categorical (X_df, categorical):
# Inputting, prevent errors for dropped columns that can't be inputted anymore
    try:  
        for category in categorical:  
        # Replace NAN in "state_construction" with the most freq value (probably GOOD)
            X_df[category] = X_df[category].fillna(X_df[category].value_counts().index[0])
            return X_df
    except KeyError as e:
        pass        

def one_hot (X_df, categorical): 
    # One hot encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    enctransform_train = enc.fit_transform(X_df[categorical])
    df = pd.concat([X_df, enctransform_train], axis = 1).drop(categorical, axis = 1)
    return df

def KNN (X_df, k = 71):
    # KNN imputation, must be done after one-hot encoding otherwise it doesnt recognize strings
    imputer = KNNImputer(missing_values = np.nan, n_neighbors=k, weights = "distance")
    X_df = imputer.fit_transform(X_df)
    return X_df

def scaler (X_df):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_df = scaler.fit_transform(X_df)
    return X_df


class Process_for_model ():
    def __init__(self, X, y, categorical):

        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42, test_size = 0.2)

        self.categorical = categorical

        # For X_train
        X_train = input_categorical (X_train, categorical)
        X_train = one_hot (X_train, categorical)
        X_train = KNN(X_train, k = 71)
        self.X_train = scaler(X_train)

        # For X_test
        X_test = input_categorical(X_test, categorical)
        X_test = one_hot(X_test, categorical)
        X_test = KNN (X_test, k = 71)
        self.X_test = scaler(X_test)
        
        # For y
        self.y_train = y_train
        self.y_test = y_test

class Process_all_dataset:
    def __init__(self, X, y, categorical):
        self.y = y
        self.categorical = categorical

        # For X_train
        X = input_categorical(X, categorical)
        X = one_hot (X, categorical)
        X = KNN(X, k = 71)
        self.X = scaler(X)





