# NYC Flood Damage Prediction (Summer 2024)

## Project Overview

This project aims to develop deep learning models to predict the damage rate (DamageRate) caused by coastal and inland flooding in New York City (NYC) based on flood depth (FloodDepth) and other relevant variables. The goal is to create two separate models, one for coastal flooding and another for inland flooding, to accurately estimate the potential damage rates in these scenarios. The deep learning models will be built using TensorFlow and scikit-learn (sklearn) libraries.

## Data

The project includes two datasets containing information on coastal and inland flooding in NYC:

1. `coastal_flooding_data.csv`: This file contains data related to coastal flooding events, including FloodDepth, DamageRate, and other potential independent variables.

2. `inland_flooding_data.csv`: This file contains data related to inland flooding events, including FloodDepth, DamageRate, and other potential independent variables.

## Approach

The project will employ deep neural networks (DNNs) built with TensorFlow to model the relationship between DamageRate (target variable) and FloodDepth (primary independent variable). Additionally, other available variables in the datasets will be explored and incorporated into the models using sklearn's feature selection and preprocessing techniques if they improve the prediction accuracy.

Two separate deep learning models will be developed:

1. **Coastal Flooding Model**: This model will be trained on the `coastal_flooding_data.csv` dataset to predict the damage rate caused by coastal flooding events based on flood depth and other relevant variables.

2. **Inland Flooding Model**: This model will be trained on the `inland_flooding_data.csv` dataset to predict the damage rate caused by inland flooding events based on flood depth and other relevant variables.

## Result
The deep neural networks were able to achieve a mean squared error (MSE) of 0.002 feet (approximately 0.024 inches) for both data sets.
