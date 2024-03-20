from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score
import pandas as pd




class metrics:
    """Child class used in each model for calculating metrics (R2, RMSE, MAE) as its attributes.
    Takes y train, y test, y predicted from train set and y predicted from test set."""
    def __init__(self, y_train, y_pred_train, y_test, y_pred, model, X_train, X_test) -> None:

        # Doing a cross validation separately, didn't have time to concatenate the datasets 
        self.scores_r2_train = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
        self.scores_r2_test = cross_val_score(model, X_test, y_test, scoring='r2', cv=2)     
        
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
        linear_regression = LinearRegression()
        linear_regression.fit(X_train,y_train)

        # Makes predictions
        y_pred = linear_regression.predict(X_test)
        y_pred_train = linear_regression.predict(X_train)

        # Stores coeficients and intercept as attributes
        self.coef = linear_regression.coef_
        self.columns = columns
        self.intercept = linear_regression.intercept_
        self.model = linear_regression

        # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred, linear_regression, X_train, X_test)

    def print_croos_r2 (self, i):
            print(f"Linear Regression {i}")
            print(self.scores_r2_train)
            print(self.scores_r2_test)
        

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
        polynomial_regressor=Pipeline(steps = [
            ("pf", PolynomialFeatures(degree)),
            ("lr",LinearRegression())]
        )
        polynomial_regressor.fit(X_train,y_train)

        # Make predictions
        y_pred = polynomial_regressor.predict(X_test)
        y_pred_train = polynomial_regressor.predict(X_train)

        # Stores coeficients and intercept as attributes
        self.coef = polynomial_regressor["lr"].coef_
        self.intercept = polynomial_regressor["lr"].intercept_
        self.model = polynomial_regressor

         # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred, polynomial_regressor, X_train, X_test)

    def print_croos_r2 (self, i):
        print(f"Polynomial Regression {i}")
        print(self.scores_r2_train)
        print(self.scores_r2_test)


class Random_forest_reg (metrics):
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        """Random forest regression: model training, model predicting and metrics.
        Number of trees and samples per leaves are set inside the class (70 trees, 2 samples per leave)."""
        # Parameters for model, those where tested before with loops to find an optimized range
        trees = 70
        samples = 2

        # Regression
        random_forest_regressor = RandomForestRegressor(n_estimators = trees, min_samples_leaf= samples)
        random_forest_regressor.fit(X_train,y_train)

        # Make predictions
        y_pred= random_forest_regressor.predict(X_test)
        y_pred_train= random_forest_regressor.predict(X_train)

        # Stores metrics as attributes
        super().__init__(y_train, y_pred_train, y_test, y_pred, random_forest_regressor, X_train, X_test)

        self.model = random_forest_regressor

        def print_croos_r2 (self, i):
            print(f"Random Forest Regression {i}")
            print(self.scores_r2_train)
            print(self.scores_r2_test)




