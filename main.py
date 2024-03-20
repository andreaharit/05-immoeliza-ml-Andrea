import pandas as pd
from preprocessing import Preprocessing
from models import Linear_reg, Poly_reg, Random_forest_reg
import joblib


 

def main():

    group_columns_drop = [["construction_year"], ["construction_year", "state_construction"]]
    group_categorical = [["district", "state_construction"], ["district"]] 


    file = "./data/cleaned_houses.csv"
    
    for i in range(2):
        columns_drop = group_columns_drop[i]
        categorical = group_categorical[i]


        prepro = Preprocessing(file = file, drop = columns_drop, cat=categorical)
        y = prepro.y

        linear = Linear_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test, columns = prepro.columns)
        linear_metrics = ["linear", linear.r2_train, linear.r2_test, linear.rmse_train, linear.rmse_test, linear.mae_train, linear.mae_test]
        out_file_linear = './models_pickle/linear_' + str(i) + '.pkl'
        joblib.dump(linear, out_file_linear)

        poly = Poly_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test)
        poly_metrics = ["polynomial", poly.r2_train, poly.r2_test, poly.rmse_train, poly.rmse_test, poly.mae_train, poly.mae_test]
        out_file_poly = './models_pickle/polynomial_' + str(i) + '.pkl'
        joblib.dump(poly, out_file_poly)

        rd_forest = Random_forest_reg(X_train= prepro.X_train, X_test= prepro.X_test, y_train= prepro.y_train, y_test=prepro.y_test)
        forest_metrics = ["random_forest", rd_forest.r2_train, rd_forest.r2_test, rd_forest.rmse_train, rd_forest.rmse_test, rd_forest.mae_train, rd_forest.mae_test]
        out_file_forest = './models_pickle/forest_' + str(i) + '.pkl'
        joblib.dump(rd_forest, out_file_forest)

        columns = ["model", "r2_train", "r2_test", "rmse_train", "rmse_test", "mae_train", "mae_test"]
        data = [linear_metrics, poly_metrics, forest_metrics]   
        
        
        print(f"Data mean {round(pd.Series(y).mean(),2)}")
        print(f"Data variance {"{:0.2e}".format(pd.Series(y).var())}")
        print(f"Data std {round(pd.Series(y).std(),2)}")
        print(f"Dropped columns: {columns_drop}")
        
        df_metrics = pd.DataFrame(data, columns = columns)  
        print(df_metrics)




if __name__ == '__main__':
    main()
