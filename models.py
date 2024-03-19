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


        self.r2_train = round(r2_score(y_train, y_pred_train),5)
        self.r2_test = round(r2_score(y_test, y_pred),5)

        #RMSE
        self.rmse_train = round(root_mean_squared_error(y_train, y_pred_train),2)
        self.rmse_test = round(root_mean_squared_error(y_test, y_pred),2)

        #MAE (mean absolute error) 
        self.mae_train = round(mean_absolute_error(y_train, y_pred_train),2)
        self.mae_test = round(mean_absolute_error(y_test, y_pred),2)

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


        self.r2_train = round(r2_score(y_train, y_pred_train),5)
        self.r2_test = round(r2_score(y_test, y_pred),5)

        #RMSE
        self.rmse_train = round(root_mean_squared_error(y_train, y_pred_train),2)
        self.rmse_test = round(root_mean_squared_error(y_test, y_pred),2)

        #MAE (mean absolute error) 
        self.mae_train = round(mean_absolute_error(y_train, y_pred_train),2)
        self.mae_test = round(mean_absolute_error(y_test, y_pred),2)

class Random_forest_reg:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        trees = 70
        leaves = 2

        treeregressor = RandomForestRegressor(n_estimators = trees, min_samples_leaf= leaves)
        treeregressor.fit(X_train,y_train)
        y_pred= treeregressor.predict(X_test)
        y_pred_train= treeregressor.predict(X_train)


        self.r2_train = round(r2_score(y_train, y_pred_train),5)
        self.r2_test = round(r2_score(y_test, y_pred),5)

        #RMSE
        self.rmse_train = round(root_mean_squared_error(y_train, y_pred_train),2)
        self.rmse_test = round(root_mean_squared_error(y_test, y_pred),2)

        #MAE (mean absolute error) 
        self.mae_train = round(mean_absolute_error(y_train, y_pred_train),2)
        self.mae_test = round(mean_absolute_error(y_test, y_pred),2)