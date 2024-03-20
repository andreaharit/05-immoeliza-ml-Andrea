# Belgian house market ML models 📈

## Project context 📝

This is the third part of a project that aims to create a machine learning model to predict the selling price of houses in Belgium.

Previous stages were:

- Scrapping [Immoweb](https://www.immoweb.be/). See [repository](https://github.com/niels-demeyer/immo-eliza-scraping-scrapegoat).
- Analysing the data for insights. See [repository](https://github.com/Yanina-Andriienko/immo-eliza-scrapeGOATS-analysis).

And now, we are building and evaluating machine learning regression models for price prediction.

## Table of Contents

- [Data 📚](#Data-📚)
- [Prepossessing details 🧹](#Prepossessing-details-🧹)
- [Models details 🤖](#Models-details-🤖)
- [Performance 🎯](#Performance-🎯)
- [Limitations 🚧](#Limitations-🚧)
- [File structure 🗂️](#Limitations-🚧)
- [Usage 🛠️](#Limitations-🚧)
- [Timeline 📅](#timeline-📅)

## Data 📚

The final processed dataset contains 18529 properties, which were scrapped and treated on February 2024.

The target variable is:
- 'price': price of the house in euros.

The features used on the final comparison between different models are:

- 'district' (categorical): Belgian province where a house is located.
- 'area_total' (numerical): total area of the lot in sqm.
- 'epc' (categorical, converted to numeric): Energy Performance Certificate of the house (A to G, where A is the best grade). 
- 'state_construction' (categorical): classification if the building needs or not improvments (GOOD, AS_NEW, TO_RENOVATE, TO_BE_DONE_UP, JUST_RENOVATED, TO_RESTORE).
- 'living_area'(numerical): total living area in sqm.
- 'livingroom_surface' (numerical): livingroom area in sqm. 
- 'kitchen_surface' (numerical): kitchen area in sqm.
- 'bedrooms' (numerical): number of bedrooms available.
- 'bathrooms' (numerical): number of bedrooms available.
- 'facades' (numerical): exposed facades of the house (1 to 4).
- 'has_garden (numerical)': boolean 0 (doesn't have) or 1 (have one).
- 'kitchen' (numerical, boolean): 0 (no equipped kitchen) or 1 (some level of equipped kitchen).
- 'has_terrace'(numerical, boolean): 0 (doesn't have) or 1 (have).
- 'has_attic'(numerical, boolean): 0 (doesn't have) or 1 (have).
- 'has_basement' (numerical, boolean): 0 (doesn't have) or 1 (have).

## Prepossessing details 🧹

Inputing was done in "livingroom_surface", "kitchen_surface" by applying an average % of their size relative to the living area.
For missing values in "epc", "facades", it was used k nearest neighbors. For "state_construction", the most frequent value.
For categorical ('district', 'state_construction') data that was not straighforth to rank without introducing bias, it was appliyed one hot encoding.


## Models details 🤖

A linear, polynomial and random forest regression were tested.
At the end, the polynomial regression required to drop an extra column ("state_construction"), for it to have meanigfull metrics.
But by dropping such a column, other models were hurt. This is note of further investigation.


## Performance 🎯

Model evaluation was done via randomly sampling 20% of the data for test, and 80% for training.




## Limitations 🚧

The model is only fitted for houses in belgium according to the subcategorization followed by Immoweb. Therefore other types of properties as appartments, or subtypes as chalets, farmhouses etc. were not considered for this model.


## File structure 🗂️

├── data
│   ├── scapegoats.csv
│   └── scapegoats.csv
├── cleaning.py
├── preprocessing.py
├── models.py
├── main.py
└── MODELSCARD.md

- scapegoats.csv is the raw data set pre-cleaning
- cleaned_houses.csv is the post-cleaning csv file
- cleaning.py is the python file that uses scapegoats.csv and outputs cleaned_houses.csv
- preprossessing.py is the python file for preprocessing the data (inputting, encoding, scaling)
- models.py is the python file containing classes for each model and a class for metrics
- main.py is the python file that runs the preprocess and models classes and prints the metrics for each model.


## Usage 🛠️




## Timeline 📅

This project took 5 days to be completed.
