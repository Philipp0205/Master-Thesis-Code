import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import scienceplots


def rfr_test():
    dataset = pd.read_csv('data/Position_Salaries.csv')

    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X, y)

    y_pred = regressor.predict([[6.5]])

    # higher resolution graph
    # X_grid = np.arange(min(X), max(X), 0.01)
    # X_grid = X_grid.reshape(len(X_grid), 1)
    #
    # plt.scatter(X, y, color='red')  # plotting real points
    # plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # plotting for predict points
    #
    # plt.title("Truth or Bluff(Random Forest - Smooth)")
    # plt.xlabel('Position level')
    # plt.ylabel('Salary')
    # plt.savefig('results/Random_Forest_Regression_100_trees.png')


def random_forest_regressor_default(X_train, y_train, X_test, y_test, train_split):
    # Create a random forest regressor
    # regressor = RandomForestRegressor(n_estimators=10, random_state=0, criterion='squared_error', oob_score=True)
    regressor = RandomForestRegressor(random_state=0, criterion='squared_error', oob_score=True)

    # Train the regressor
    regressor.fit(X_train, y_train)

    # Caluclate feature imporances.
    calculate_feature_importances(regressor, X_train)

    #
    y_pred = regressor.predict(X_test)

    # Calculate mean squared error and root mean squared error
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'mse: {round(mse, 3)}')
    print(f'rmse: {round(rmse, 3)}')

    oop = regressor.oob_score_

    params = regressor.get_params()
    description = f'{params["n_estimators"]} estimators, oop: {round(oop, 2)}'

    # Create a scatter plot for the prediction
    create_predict_scatter('results/rfr_default/', 'Random_Forest_Regression', y_test, y_pred, description, train_split)


def calculate_feature_importances(regressor, X_train):
    # Importances
    importances = regressor.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_train.columns[i] for i in indices]

    # Use scientific style
    plt.style.use(['science', 'bright'])

    # Create plot
    plt.figure(dpi=600)
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(X_train.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_train.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(f'results/Random_Forest_Regression_importances.png')
    plt.clf()


# Prepate the data for the random forst regression
# Split data into tain and test set
def rfr_prepare_data(input_directory, train_split):
    # Get consolidated csv file
    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')

    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # Choose as testing set all values with an die opening of 20mm
    X_test = X.loc[X['die_opening'] == train_split]
    y_test = y.loc[X['die_opening'] == train_split]

    X_train = X[X['die_opening'] != train_split]
    y_train = y[X['die_opening'] != train_split]

    return X_train, y_train, X_test, y_test


def random_forest_ada_boost(X_train, y_train, X_test, y_test, train_split):
    # adaboost = AdaBoostRegressor(n_estimators=10, random_state=0)
    adaboost = AdaBoostRegressor(random_state=0)
    adaboost.fit(X_train, y_train)

    # Predict test set
    y_pred = adaboost.predict(X_test)

    # Calculate mse and rmse
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'mse: {round(mse, 3)}')
    print(f'rmse: {round(rmse, 3)}')

    params = adaboost.get_params()
    description = f'{params["n_estimators"]} estimators'
    # Create Scatter plot for the prediction
    create_predict_scatter('results/rfr_boost/', 'Adaboost_Regression', y_test, y_pred, description, train_split)


def create_predict_scatter(output_path, name, y_test, y_pred, description, train_split):
    # Calculate mse and rmse
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'mse: {round(mse, 3)}')
    print(f'rmse: {round(rmse, 3)}')

    # Create scatter plot for results
    plt.style.use(['science', 'scatter', 'bright', 'grid'])

    plt.scatter(X_test['distance'].values, y_test)
    plt.scatter(X_test['distance'].values, y_pred)

    plt.title(
        f"({description})",
        fontsize=8)
    plt.suptitle(f'{name}, mse: {round(mse, 2)}, remse: {round(rmse, 2)}, train split: {train_split}',
                 fontsize=8)
    plt.xlabel('RM')
    plt.ylabel('Price')
    plt.savefig(f'{output_path}{name}{train_split}.png')
    plt.clf()


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    train_split = 30
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    X_train, y_train, X_test, y_test = rfr_prepare_data(input_directory, train_split)
    random_forest_regressor_default(X_train, y_train, X_test, y_test, train_split)
    random_forest_ada_boost(X_train, y_train, X_test, y_test, train_split)
