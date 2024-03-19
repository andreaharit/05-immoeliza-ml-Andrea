from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


class Linear_reg:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:        

        regressor = LinearRegression()
        regressor.fit(X_train,y_train)

        # Make predictions using the testing set
        y_pred = regressor.predict(X_test)
        y_pred_train = regressor.predict(X_train)

        self.coef = regressor.coef_
        self.intercept = regressor.intercept_
        self.r2_train = r2_score(y_train, y_pred_train)
        self.r2_test = r2_score(y_test, y_pred)

        #RMSE
        self.rmse_train = root_mean_squared_error(y_train, y_pred_train)
        self.rmse_test = root_mean_squared_error(y_test, y_pred)

        #MAE (mean absolute error) 
        self.mae_train = mean_absolute_error(y_train, y_pred_train)
        self.mae_test = mean_absolute_error(y_test, y_pred)

class Poly_reg:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:        

        degree = 2
        polyreg=Pipeline(steps = [
            ("pf", PolynomialFeatures(degree)),
            ("lr",LinearRegression())]
        )

        polyreg.fit(X_train,y_train)
        y_pred = polyreg.predict(X_test)
        y_pred_train= polyreg.predict(X_train)


        # Make predictions using the testing set
        y_pred = polyreg.predict(X_test)
        y_pred_train = polyreg.predict(X_train)

        self.coef = polyreg["lr"].coef_
        self.intercept = polyreg["lr"].intercept_
        self.r2_train = r2_score(y_train, y_pred_train)
        self.r2_test = r2_score(y_test, y_pred)

        #RMSE
        self.rmse_train = root_mean_squared_error(y_train, y_pred_train)
        self.rmse_test = root_mean_squared_error(y_test, y_pred)

        #MAE (mean absolute error) 
        self.mae_train = mean_absolute_error(y_train, y_pred_train)
        self.mae_test = mean_absolute_error(y_test, y_pred)

class Random_forest_reg:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        n = 100
        j = 4

        treeregressor = RandomForestRegressor(n_estimators = n, min_samples_leaf= j)
        treeregressor.fit(X_train,y_train)
        y_pred= treeregressor.predict(X_test)
        y_pred_train= treeregressor.predict(X_train)


        self.r2_train = r2_score(y_train, y_pred_train)
        self.r2_test = r2_score(y_test, y_pred)

        #RMSE
        self.rmse_train = root_mean_squared_error(y_train, y_pred_train)
        self.rmse_test = root_mean_squared_error(y_test, y_pred)

        #MAE (mean absolute error) 
        self.mae_train = mean_absolute_error(y_train, y_pred_train)
        self.mae_test = mean_absolute_error(y_test, y_pred)
