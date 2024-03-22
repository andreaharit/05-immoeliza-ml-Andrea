import pandas as pd
from preprocessing import Process_for_model, Process_all_dataset
from models import Random_forest_reg
import joblib

from sklearn.model_selection import KFold, cross_validate


def main():

    categorical = ["district","state_construction"]
    file = "./data/cleaned_houses.csv"

    df = pd.read_csv(file)
    # Experimental dropping
    #df.drop([], axis=1, inplace=True)
    
    # Shuffle DF because its ordered per price, and this breaks cross validation
    df = df.sample(frac=1).reset_index(drop=True)
    

    # Split in feature/target and training set/test set
    X = df.drop(columns=["price"])
    y = df['price']     

    prepro_split = Process_for_model(X = X, y = y, categorical= categorical)

    X_train = prepro_split.X_train 
    X_test = prepro_split.X_test
    y_train = prepro_split.y_train
    y_test = prepro_split.y_test
    columns_onehot =prepro_split.columns_onehot


    prepro_all = Process_all_dataset(X = X, y = y, categorical= categorical)
    y = prepro_all.y
    X = prepro_all.X



    rd_forest = Random_forest_reg(X_train= X_train, X_test= X_test, y_train= y_train, y_test= y_test)
    forest_metrics = rd_forest.metrics   



    columns = ["model", "r2_train", "r2_test", "rmse_train", "rmse_test", "mae_train", "mae_test"]
    data = forest_metrics
    
    
    print(f"Data mean {round(pd.Series(y).mean(),2)}")
    print(f"Data variance {"{:0.2e}".format(pd.Series(y).var())}")
    print(f"Data std {round(pd.Series(y).std(),2)}")
    
    df_metrics = pd.DataFrame([data], columns = columns)  

    print(df_metrics)


    # Doing a cross validation  
      
    scores = ('r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error')
    cross_validation = cross_validate(rd_forest.model, X, y, scoring= scores, cv=5)
    print (f"Cross validation, mean R2: {round(abs(cross_validation["test_r2"].mean()),2)}")
    print (f"Cross validation, mean RMSE: {round(abs(cross_validation['test_neg_root_mean_squared_error'].mean()),2)}")
    print (f"Cross validation, mean MAE: {round(abs(cross_validation['test_neg_mean_absolute_error'].mean()),2)}")


    print (columns_onehot)

    out_file_forest = './models_pickle/forest.pkl'
    joblib.dump(rd_forest.model, out_file_forest)
    

if __name__ == '__main__':
    main()
