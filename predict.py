import pandas as pd
import json
import joblib
import warnings



def main():
    #Supress warnin "has feature names, but StandardScaler was fitted without feature names"
    #Since it doesnt affect the calculation
    
    warnings.filterwarnings("ignore")


    categorical = ["district","state_construction"]

    # Load new data
    with open('input_api.json') as f:
        data = json.load(f)   
    
    X = pd.json_normalize(data['house'])

   
    # Encodes new data with saved encoding from train set
    encoder = './models_pickle/encoding.pkl'
    loaded_enc = joblib.load(encoder)

    enctransform = loaded_enc.transform(X[categorical])
    X = pd.concat([X, enctransform], axis = 1).drop(categorical, axis = 1)

    # Scales new data with saved scaler from train set
    scaler = './models_pickle/scaler.pkl'
    loaded_scaler = joblib.load(scaler)
    X = loaded_scaler.transform(X)

    # Loads and use model to predict new price
    model = './models_pickle/forest.pkl'
    loaded_model = joblib.load(model)
   
    result = loaded_model.predict(X)
    print(result)


if __name__ == '__main__':
    main()
