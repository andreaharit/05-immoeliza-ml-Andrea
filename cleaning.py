import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def count_missing (df) -> None:
    print (df.isnull().sum() *100 / len(df)) 

def count_zeros (df) -> None:
    print(df[df == 0].count(axis=0) *100/len(df.index))

def IQR(df, column) -> float:
    """Compute lowerlimit and upperlimit and trim outliers via Tukey's IQR fence"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    iqr = Q3 - Q1
    lowerlimit = Q1 - 1.5 * iqr
    upperlimit = Q3 + 1.5 * iqr
    df = df[df[column].between(lowerlimit, upperlimit)]
    return df
    
def coef_matrix (df, columns):
    cor = df[columns].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(cor, cmap=plt.cm.CMRmap_r,annot=True)
    plt.show()  

def main():
    file = "./data/scapegoats.csv"
    df = pd.read_csv(file)

    print(f"The original dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    # Select only houses        
    mask_houses = df["subtype"] == "HOUSE"
    df= df[mask_houses]
    print(f"After selecting just houses: {df.shape[0]} rows and {df.shape[1]} columns")

    # replace empty strings with nan
    df.replace("", np.nan, inplace=True)
    
    # Cleaning outliers

    # Cleans price outliers
    # Excludes biddings (per immoweb exploration, min price of houses is 40k)
    mask_price = df["price"] > 40000
    df = df[mask_price]

    # Cleans the price outliers via IQR function
    df = IQR(df, "price")

    # Cleans total_area outliers
    # Excludes small properties that are garages under wrong input on immoweb
    mask_total_area = df["area_total"] > 50
    df = df[mask_total_area]

    # Excludes per IQR
    df = IQR(df, "area_total")

    # Cleaning living_area outliers
    # Excludes again garages and wrong inputs that fall under living_area
    mask_living = df["living_area"] > 50
    df = df[mask_living]

    # Excludes outliers per IQR
    df= IQR(df, "living_area")

    # Starts column droppping

    # Heating is too incomplete. It has only gas, doesn't show electrical for example
    df.drop(["id","subtype", "heating"], axis=1, inplace=True)
    # Better location predictior is district, other locations are too many or too little
    df.drop(["city", "postal_code", "province"], axis=1, inplace=True)
    # has_terrace has less missing values than area, same as has_garden
    df.drop(["terrace_area", "garden_area"], axis=1, inplace=True)
    # Rooms (72% missing) is covered by bathrooms and bedrooms numbers which are more complete 
    df.drop(["rooms"], axis=1, inplace=True)
    # Furnished, swimmingpool, fireplace are almost all 0. 
    # Too uneven distribution for categorical data. And have very low correlation with the price.
    df.drop(['furnished','fireplace', 'swimmingpool'], axis=1, inplace=True)

    # Drop rows where bathroom or bedroom is zero or NaN
    # They are very little of the df
    df = df[df.bedrooms != 0]
    df = df[df.bathrooms != 0]
    df.dropna(subset = ['bedrooms', 'bathrooms'], inplace= True)

    # Corrects houses with more than 4 facades, because thats weird
    df.loc[df["facades"] > 4, 'facades'] = 4
    
    # Putting Nan in Epcs incorrectly inputed
    mask_epc = df["epc"].isin(["A++", "A+", "A", "B", "C", "D", "E", "F", "G"])
    df.loc[~mask_epc, 'epc'] = np.nan


    # Group EPCS As and give them a numerical weight
    group_epc = {"A++": 6, 
                "A+": 6, 
                "A": 6,
                "B":5,
                "C":4,
                "D":3,
                "E":2,
                "F":1,
                "G": 0
                }
    df = df.replace({"epc": group_epc})


    # Puting Nan in state of construction incorrectly inputted
    mask_state = df["state_construction"].isin(["GOOD", "JUST_RENOVATED", "AS_NEW", "TO_RENOVATE", "AS_NEW","TO_RESTORE","TO_BE_DONE_UP"])
    df.loc[~mask_state, 'state_construction'] = np.nan

    # Put NaN where kitchen or livingroom area is bigger than living area
    df.loc[df["kitchen_surface"] > df["living_area"], "kitchen_surface"] = np.nan
    df.loc[df["livingroom_surface"] > df["living_area"], "living_area"] = np.nan

    # Drop rows where kitchen or livingroom area is bigger than living area
    # Unreliable rows
    mask_kitchen = df["kitchen_surface"] > df["living_area"]
    mask_living = df["livingroom_surface"] > df["living_area"]
    df = df[~mask_kitchen]
    df = df[~mask_living]

    # Drop rows where living_area is empty, they are very little
    df = df.dropna(subset = ['living_area']) 


    # substitute NaN for 0 in some columns assuming no imput means it doesn't have
    df["has_terrace"] = df["has_terrace"].fillna(0)
    df["has_basement"] = df["has_basement"].fillna(0)
    df["has_attic"] = df["has_attic"].fillna(0)
    df["has_garden"] = df["has_garden"].fillna(0)


    # Dropping duplicates
    df.drop_duplicates(inplace=True)

    # Force dtypes that make sense
    convert_dict = {'bathrooms': 'Int64',
                    "facades": 'Int64',
                    'has_garden': 'Int64',
                    "has_terrace": 'Int64',
                    'has_attic': 'Int64',
                    "has_basement": 'Int64',
                    "construction_year": 'Int64',
                    "epc":'Int64'
                    }    
    df = df.astype(convert_dict)

    # Sort df for price for easier visualization of CSV
    df = df.sort_values(by=['price'])
    df = df.reset_index(drop=True)

    print(f"After cleaning the data: {df.shape[0]} rows and {df.shape[1]} columns")
    print("The percentage of missing values is:")
    count_missing(df)
    print("The percentafe of zeros is:")
    count_zeros(df)


    # Plot coef matrix    
    columns_coef = ['price','living_area', 'bedrooms',
        'bathrooms', 'livingroom_surface', 'kitchen_surface', 'facades',
        'has_garden', 'kitchen', 'has_terrace', 'has_attic', 'has_basement',
        'construction_year','area_total']
    coef_matrix(df, columns_coef )

    # save to csv
    df.to_csv("./data/cleaned_houses.csv", index=False)


if __name__ == "__main__":
    main()