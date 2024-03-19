# Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


class Preprocessing:
    """
    Class handles all the preprocessing regarding the data:
        -Dropping columns to fine tune a model
        -Splitting data
        -One hot encoding
        -Scaling
    Args: 
        file(str): CSV to be read with pre-cleanned data
        drop(list): list of columns to be dropped for a particular model.
        cat(list): list of not dropped columns that are categorical and will be encoded
    Attributes:
        self.X = pos-dropping, pre-splitting feature columns with data
        self.y = pre-split target column with data
        self.X_train = X train set pos-processing
        self.X_test = X test set pos-processing
        self.y_test = y_test pos-processing
        self.y_train = y_train pos-processint
        self.columns = list with columns names of X_train pos-processing
    """
    def __init__(self, file, drop, cat) -> None:
        # Load clean CSV
        df = pd.read_csv(file)

        # Dropping columns that hurt the specific model
        columns_drop = drop
        categorical = cat

        df.drop(columns = columns_drop, inplace= True)

        # Split in training set and test set
        X = df.drop(columns=["price"])
        y = df['price']
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.2)

        # One hot encoding
        # For training set
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        enctransform_train = enc.fit_transform(X_train[categorical])
        X_train = pd.concat([X_train, enctransform_train], axis = 1).drop(categorical, axis = 1)

        # For testing set
        enc2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        enctransform_test = enc2.fit_transform(X_test[categorical])
        X_test = pd.concat([X_test, enctransform_test], axis = 1).drop(categorical, axis = 1)

        # Colecting colum names for printing later coefitients
        columns = X_train.columns

        # Inputting
        # Replace NAN in kitchen and living room by multiplying living area for a average %
        percent_k = X_train["kitchen_surface"].sum()/X_train["living_area"].sum() 
        percent_l = X_train["livingroom_surface"].sum()/X_train["living_area"].sum() 
        X_train['livingroom_surface'] = X_train['livingroom_surface'].fillna(round(X_train["living_area"]*percent_l,0))
        X_train['kitchen_surface'] = X_train['kitchen_surface'].fillna(round(X_train["living_area"]*percent_k,0))

        # KNN imputation
        imputer = KNNImputer(missing_values = np.nan, n_neighbors=5, weights = "distance")
        X_train = imputer.fit_transform(X_train)
        imputer = KNNImputer(n_neighbors=5)
        X_test = imputer.fit_transform(X_test)

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create class attributes
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.columns = columns
