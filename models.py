from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd




class metrics:
    """Child class used in each model for calculating metrics (R2, RMSE, MAE) as its attributes.
    Takes y train, y test, y predicted from train set and y predicted from test set."""
    def __init__(self, y_train, y_pred_train, y_test, y_pred ) -> None:
        
        # R2
        self.r2_train = round(r2_score(y_train, y_pred_train),5)
        self.r2_test = round(r2_score(y_test, y_pred),5)

        #RMSE
        self.rmse_train = round(root_mean_squared_error(y_train, y_pred_train),2)
        self.rmse_test = round(root_mean_squared_error(y_test, y_pred),2)

        #MAE (mean absolute error) 
        self.mae_train = round(mean_absolute_error(y_train, y_pred_train),2)
        self.mae_test = round(mean_absolute_error(y_test, y_pred),2)

class Linear_reg (metrics):
    """Linear regression: model training, model predicting and metrics"""
    def __init__(self, X_train, X_test, y_train, y_test, columns) -> None:        
        # Regression
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)

        # Makes predictions
        y_pred = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)

        # Stores coeficients and intercept as attributes
        self.coef = regressor.coef_
        self.columns = columns
        self.intercept = regressor.intercept_

        # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred)

    def print_coef (self):
        coef_list = pd.DataFrame(zip(self.columns, self.coef))
        coef_list.columns = ["feature", "coef"]
        most_important = coef_list.sort_values(by= "coef", ascending= False)
        pd.options.display.max_rows = 999

        print (f"Intercept: {self.intercept}")
        print (most_important)


class Poly_reg (metrics):
    """Polynomial regression: model training, model predicting and metrics.
    Number of degrees for the regression is set inside the class (2)."""

    def __init__(self, X_train, X_test, y_train, y_test) -> None: 

        # Degree parameter for polynomial regression
        degree = 2

        # Generating polynome and regressing
        polyreg=Pipeline(steps = [
            ("pf", PolynomialFeatures(degree)),
            ("lr",LinearRegression())]
        )
        polyreg.fit(X_train,y_train)

        # Make predictions
        y_pred = polyreg.predict(X_test)
        y_pred_train = polyreg.predict(X_train)

        # Stores coeficients and intercept as attributes
        self.coef = polyreg["lr"].coef_
        self.intercept = polyreg["lr"].intercept_

         # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred)


class Random_forest_reg (metrics):
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        """Random forest regression: model training, model predicting and metrics.
        Number of trees and samples per leaves are set inside the class (70 trees, 2 samples per leave)."""
        # Parameters for model, those where tested before with loops to find an optimized range
        trees = 70
        samples = 2

        # Regression
        treeregressor = RandomForestRegressor(n_estimators = trees, min_samples_leaf= samples)
        treeregressor.fit(X_train,y_train)

        # Make predictions
        y_pred= treeregressor.predict(X_test)
        y_pred_train= treeregressor.predict(X_train)

        # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred)




