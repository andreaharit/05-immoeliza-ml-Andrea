# Model card

## Project context

This is the third part of a project that aims to create a machine learning model to predict the selling price of houses in Belgium.

Previous stages were:

- Scrapping Immoweb to gather the raw data. See repository (insert link)
- Analysing the data for insights. See repository (insert link)

And now, here, we are building and evaluating machine learning regression models for price prediction.


## Data

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

## Prepossessing details

Inputing was done in "livingroom_surface", "kitchen_surface" by applying an average % over the living area.
For missing values in "epc", "facades", "state_construction", it was used k nearest neighbors.
For categorical ('district', 'state_construction') data that was not straighforth to rank without introducing bias, it was applyed one hot encoding.


## Models details

A linear, polynomial and random forest regression were tested.
At the end, the polynomial regression required to drop an extra column ("state_construction"), for it to have meanigfull metrics.
But by dropping such a column, other models were hurt. This is note of further investigation.


## Performance

Model evaluation was done via randomly sampling 20% of the data for test, and 80% for training.


PUT THE METRICS WITH AND WITHOUT THE STATE OF CONSTRUCTION

## Limitations

TALK ABOUT HOW THE MODEL IS WEIRD WITH COLUM SELECTION, HOW RANDOM TREE IS TOUCHY FEELY.

## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

## Maintainers

Who to contact in case of questions or issues?

## File structure