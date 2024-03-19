import pandas as pd
from preprocessing import Preprocessing
from models import Linear_reg, Poly_reg, Random_forest_reg
 

def main():

    file = "./data/cleaned_houses.csv"
    columns_drop = ["construction_year", "state_construction"]
    categorical = ["district"]   

    prepro = Preprocessing(file = file, drop = columns_drop, cat=categorical)
    y = prepro.y

    linear = Linear_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test)
    linear_metrics = ["linear", linear.r2_train, linear.r2_test, linear.rmse_train, linear.rmse_test, linear.mae_train, linear.mae_test]


    poly = Poly_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test)
    poly_metrics = ["polynomial", poly.r2_train, poly.r2_test, poly.rmse_train, poly.rmse_test, poly.mae_train, poly.mae_test]

    rd_forest = Random_forest_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test)
    forest_metrics = ["random_forest", rd_forest.r2_train, rd_forest.r2_test, rd_forest.rmse_train, rd_forest.rmse_test, rd_forest.mae_train, rd_forest.mae_test]

    columns = ["regression", "r2_train", "r2_test", "rmse_train", "rmse_test", "mae_train", "mae_test"]
    data = [linear_metrics, poly_metrics, forest_metrics]

    
    
    df_metrics = pd.DataFrame(data, columns = columns)

    print(df_metrics)

if __name__ == '__main__':
    main()
