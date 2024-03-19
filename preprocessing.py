# Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, PredictionErrorDisplay
from sklearn.linear_model import LinearRegression


class Preprocessing:
    def __init__(self, file, drop, cat) -> None:
        # Load clean CSV
        df = pd.read_csv(file)
        # Experimental column dropping
        # Construction year hurt the model by 2%, lots of missing data, maybe badly inputed
        # and likely covered by state_construction, size, epc already
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
        # Replace NAN in kitchen and living room by multiplying living area for a %
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

        # Class objects
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.columns = columns


file = "./data/cleaned_houses.csv"
columns_drop = ["construction_year"]
categorical = ["district", "state_construction"]

prepro = Preprocessing(file = file, drop = columns_drop, cat=categorical)
regressor = LinearRegression()
regressor.fit(prepro.X_train,prepro.y_train)


# Make predictions using the testing set
y_pred = regressor.predict(prepro.X_test)
y_pred_train = regressor.predict(prepro.X_train)

#DATA
print(f"mean {pd.Series(prepro.y).mean()}")
print(f"var {"{:e}".format(pd.Series(prepro.y).var())}")
print(f"std {pd.Series(prepro.y).std()}")
print()

#R2
print(f"R2 train: {r2_score(prepro.y_train, y_pred_train)}")
print(f"R2 test: {r2_score(prepro.y_test, y_pred)}")
print()
