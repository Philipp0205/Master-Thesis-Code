import pickle
import random

import random_forest
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline

# import leave one out cross validation
from sklearn.model_selection import LeaveOneOut, KFold


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


def random_forest(X_train, y_train, X_test, y_test, train_split):
    # Create random forest regressors with different numbers of trees
    regr_1 = RandomForestRegressor(random_state=0, n_estimators=1)
    regr_100 = RandomForestRegressor(random_state=0, n_estimators=100)
    reg_200 = RandomForestRegressor(random_state=0, n_estimators=200)

    # Train the regressors
    regr_1.fit(X_train, y_train)
    regr_100.fit(X_train, y_train)
    reg_200.fit(X_train, y_train)

    # Predict on new data
    y_pred_1 = regr_1.predict(X_test)
    y_pred_100 = regr_100.predict(X_test)
    y_pred_200 = reg_200.predict(X_test)

    # Create scatters for all predictions
    create_predict_scatter('results/rfr_default/', 'Random_Forest_Regression_1_tree', y_test,
                           y_pred_1,
                           'Random Forest 1 Tree',
                           train_split)
    create_predict_scatter('results/rfr_default/', 'Random_Forest_Regression_100_trees', y_test,
                           y_pred_100,
                           'Random Forest 100 Trees', train_split)
    create_predict_scatter('results/rfr_default/', 'Random_Forest_Regression_200_trees', y_test,
                           y_pred_200,
                           'Random Forest 200 Trees', train_split)


def random_forest_default(X_train, y_train, X_test, y_test, train_split):
    # Create a random forest regressor
    # regressor = RandomForestRegressor(n_estimators=10, random_state=0, criterion='squared_error', oob_score=True)
    regressor_default = RandomForestRegressor(random_state=0, criterion='squared_error',
                                              oob_score=True)
    regressor_100 = RandomForestRegressor(random_state=0, criterion='squared_error',
                                          n_estimators=200)

    # Train the regressor
    regressor_default.fit(X_train, y_train)
    regressor_100.fit(X_train, y_train)

    # Caluclate feature imporances.
    calculate_feature_importances(regressor_default, X_train)

    y_pred = regressor_default.predict(X_test)

    parameters = {
        'n_estimators': [1, 10, 100, 1000],
        'max_depth': [1, 10, 100, 1000],
    }
    regr = RandomForestRegressor(random_state=0, n_estimators=1, criterion='squared_error')

    clf = GridSearchCV(regr, parameters)

    clf.fit(X_train, y_train)

    oop = regressor_default.oob_score_
    clf.fit(X_train, y_train)
    regr.fit(X_train, y_train)

    y_pred2 = clf.predict(X_test)
    y_pred3 = regr.predict(X_test)
    y_pred4 = regressor_100.predict(X_test)

    # Print n_estimators
    print(f'Default Regressor parameters: {regressor_default.get_params()["n_estimators"]}')
    print("Tuned Random Forest Parameters: {}".format(clf.best_params_))
    print(f"Tuned Random Forest Parameters 100: {regressor_100.get_params()['n_estimators']}")

    params = regressor_default.get_params()
    description = f'Random Forest Default'
    create_predict_scatter('results/rfr_default/', 'Random_Forest_Regression', y_test, y_pred,
                           description, train_split)

    description = f'Random Forest Tuned'
    create_predict_scatter('results/rfr_default/', 'Random Forest Regression TUNED', y_test,
                           y_pred2, description,
                           train_split)

    create_predict_scatter('results/rfr_default/', 'Random Forest Regression TUNED2', y_test,
                           y_pred3, description,
                           train_split)

    create_predict_scatter('results/rfr_default/', 'Random Forest Regression TUNED100', y_test,
                           y_pred3, description,
                           train_split)


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
    plt.savefig(f'results/rfr_default/Random_Forest_Regression_importances.png')
    plt.clf()


# Prepate the data for the random forst regression
# Split data into tain and test set
def non_random_split(df, train_split):
    correlations(df)

    print(f'Number of samples: {len(df.index)}')

    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # Choose as testing set all values with an die opening of 20mm
    X_test = X.loc[X['die_opening'] == train_split]
    y_test = y.loc[X['die_opening'] == train_split]

    X_train = X[X['die_opening'] != train_split]
    y_train = y[X['die_opening'] != train_split]

    return X, y, X_train, y_train, X_test, y_test


def random_forest_ada_boost(X_train, y_train, X_test, y_test, train_split):
    # adaboost = AdaBoostRegressor(n_estimators=10, random_state=0)
    adaboost = AdaBoostRegressor()
    adaboost.fit(X_train, y_train)

    # Predict test set
    y_pred = adaboost.predict(X_test)

    # Calculate mse and rmse
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # print(f'mse: {round(mse, 3)}')
    # print(f'rmse: {round(rmse, 3)}')

    params = adaboost.get_params()
    description = f'{params["n_estimators"]} estimators'
    # Create Scatter plot for the prediction
    create_predict_scatter('results/rfr_boost/', 'Adaboost_Regression', y_test, y_pred, description,
                           train_split)


def create_predict_scatter(output_path, name, y_test, y_pred, description, train_split):
    # Calculate mse and rmse
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'{name}')
    print(f'mse: {round(mse, 5)}')
    print(f'rmse: {round(rmse, 5)}')

    # Create scatter plot for results
    plt.style.use(['science', 'scatter', 'bright', 'grid'])

    plt.scatter(X_test['distance'].values, y_test)
    plt.scatter(X_test['distance'].values, y_pred)

    plt.title(
        f"({description})",
        fontsize=8)
    plt.suptitle(
        f'{name}, mse: {round(mse, 2)}, remse: {round(rmse, 2)}, train split: {train_split}',
        fontsize=8)
    plt.xlabel('RM')
    plt.ylabel('Price')
    plt.savefig(f'{output_path}{name}{train_split}_new.png')
    plt.clf()


def gradient_boosting_regressor(X, y, X_train, y_train, X_test, y_test, train_split):
    # steps = [
    #     ('scale', StandardScaler()),
    # ('GBR', GradientBoostingRegressor(n_estimators=500, learning_rate=0.03))
    # ]

    gbr_default = GradientBoostingRegressor().fit(X_train, y_train)

    # Model
    gbr = GradientBoostingRegressor(n_estimators=1071, learning_rate=0.3).fit(X_train, y_train)

    # Predict
    y_pred = gbr.predict(X_test)
    y_pred_default = gbr_default.predict(X_test)

    # RMSE of the predictions
    print(f'RMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)}')
    print(f'RMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred_default)), 4)}')

    errors = [mean_squared_error(y_test, preds)
              for preds in gbr.staged_predict(X_test)]

    find_best_n_estimators(errors, "gradient_boosting")


def find_best_n_estimators(errors, name):
    # Loop for the best number
    best_n_estimators = np.argmin(errors) + 1

    # Crate line plot for the prediction with matplotlib
    plt.style.use(['science', 'bright', 'grid'])
    plt.plot(range(1071), errors)
    plt.title(f'Best number of estimators at {best_n_estimators}')
    plt.xlabel('Number of estimators')
    plt.ylabel('MSE')
    plt.savefig(f'results/{name}_best_n_estimators.png')


def correlations(df):
    print('Plotting Correlations')
    # Create correlation matrix
    df.corr().style.background_gradient(cmap='coolwarm')

    # Plot heatmap
    plt.figure(dpi=600)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('results/correlation_matrix.png')
    plt.clf()


def preprocessing():
    # Create pipeline and scale the data
    pipe = Pipeline([("scaler", MinMaxScaler()), ("random_forest", RandomForestRegressor())])

    # Fit the pipeline
    # pipe.fit(X_train, y_train)

    # Predict
    # print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

    return pipe


def grid_search(pipe, X_train, y_train, X_test, y_test):
    print('----------- Grid Search -----------')
    # Parameters for grid search random forest regressor
    param_grid = {'rfr__n_estimators': [1, 2, 3, 10, 20, 30, 31, 32, 33, 34, 35, 100],
                  'rfr__max_depth': [1, 2, 3, 4, 5, 10, 20, 30, 100],
                  'rfr__min_samples_split': [2, 4],
                  'rfr__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ],
                  'rfr__bootstrap': [True, False],
                  'rfr__criterion': ['squared_error', 'absolute_error'],
                  }

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=1)
    grid.fit(X_train, y_train)

    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    print("R2 score: {:.2f}".format(r2_score(y_test, grid.predict(X_test))))
    print("Best parameters: {}".format(grid.best_params_))


def random_test_train_split(input_directory):
    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')

    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, y_train, X_test, y_test


def rfr_with_grid_search():
    print('------- Non random test train split ------')
    X, y, X_train, y_train, X_test, y_test = non_random_split(input_directory, train_split)
    pipe2 = preprocessing()
    grid_search(pipe2, X_train, y_train, X_test, y_test)


def rfr_final_after_grid_search(name, X_train, y_train, X_test, y_test):
    # Best parameters: {'rfr__bootstrap': True, 'rfr__criterion': 'absolute_error', 'rfr__max_depth': 30,
    # 'rfr__min_samples_leaf': 2, 'rfr__min_samples_split': 4, 'rfr__n_estimators': 10}
    print(f'------- {name} ------')
    print(f'Number of training samples: {len(X_train)}')
    print(f'Number of testing samples: {len(X_test)}')
    print('-----')
    pipe = Pipeline(
        [("scaler", MinMaxScaler()),
         ("random_forest", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                       max_depth=30, min_samples_leaf=2,
                                       min_samples_split=4, n_estimators=10))])
    sample = X_test.iloc[0]
    start = time.time()
    pipe.fit(X_train, y_train)
    stop = time.time()
    s = stop - start
    print(f"Training time: {s} s")

    sample = X_test.iloc[0]
    start = time.time()
    pipe.predict([sample])
    stop = time.time()
    ms = (stop - start) * 1000

    print(f"Prediction time (runtime): {ms} ms")

    save_trained_model(pipe, name)

    y_pred = pipe.predict(X_test)
    print("MAE: {:.3f}".format(mean_absolute_error(y_test, y_pred)))
    print("MSE: {:.3f}".format(mean_squared_error(y_test, y_pred)))
    print("RMSE: {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("R2 score: {:.3f}".format(r2_score(y_test, y_pred)))
    print('-----------------')

    return pipe


def calculate_variance_of_cross_validation(X, y, model):
    scores = cross_val_score(model, X, y, cv=5)

    variance_of_cv = np.var(scores, ddof=1)
    print(f'Variance of cross validation: {round(variance_of_cv, 3)}')


def leave_one_out_cross_validation(X, y):
    pipe = Pipeline(
        [("scaler", MinMaxScaler()),
         ("random_forest", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                       max_depth=30, min_samples_leaf=2,
                                       min_samples_split=4, n_estimators=10))])

    scores = []

    loo = LeaveOneOut()

    # Use the LOOCV object to get the training and test indices for each iteration
    for train_index, test_index in loo.split(X):
        # Get the training and test data for this iteration
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the regressor to the training data and predict on the test data
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Calculate the performance score and store it
        score = np.mean((y_test - y_pred) ** 2)

        # Append the evaluation score to the scores list
        scores.append(score)

    # Calculate the mean performance score
    mean_score = np.mean(scores)
    # Variance score
    variance_score = np.var(scores)

    print(f'Leave one out cross validation mean: {round(mean_score, 3)}')
    print(f'Variance of leave one out cross validation: {round(variance_score, 3)}')


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


# Save trained model to disk
def save_trained_model(model, name):
    project_root = get_project_root()
    output_directory = project_root / 'learner' / 'random_forest' / 'saved_models'

    with open(f'{output_directory}/{name}.model', 'wb') as f:
        pickle.dump(model, f)


def missing_values(df, number_of_missing_values):
    print('------- Missing values: Removing vt combinations --------')
    # Iterate through all rows and get all thickness die_opening combinations
    thickness_die_opening_combinations = []

    for index, row in df.iterrows():
        thickness_die_opening_combinations.append([row['thickness'], row['die_opening']])

    # Remove duplicates
    thickness_die_opening_combinations = list(
        set(tuple(x) for x in thickness_die_opening_combinations))
    number_of_combinations = len(thickness_die_opening_combinations)

    # Select two numbers between 0 and number_of_combinations
    random_numbers = random.sample(range(0, number_of_combinations), number_of_missing_values)

    # Get the rows where the index are the random numbers
    rows_to_remove = []
    for i in random_numbers:
        rows_to_remove.append(thickness_die_opening_combinations[i])

    # Remove rows from dataframe
    remaining_data = df[~df[['thickness', 'die_opening']].apply(tuple, 1).isin(rows_to_remove)]

    # Get removed rows
    removed_data = df[df[['thickness', 'die_opening']].apply(tuple, 1).isin(rows_to_remove)]

    # print combinations which got removed
    for i in range(0, len(random_numbers)):
        print(f'Removed: {thickness_die_opening_combinations[i]}')

    return removed_data, remaining_data


if __name__ == '__main__':
    train_split = 30
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')
    X, y, X_train, y_train, X_test, y_test = non_random_split(df, train_split)

    missing_values(X_train, 2)

    # random_test_train_split(input_directory)
    # pipe = rfr_final_after_grid_search('NON Random Test Train Split ', X_train, y_train, X_test, y_test)

    # calculate_variance_of_cross_validation(X, y, pipe)

    # leave_one_out_cross_validation(X, y)

    # Archive

    # X, y_train, X_test, y_test = random_test_train_split(input_directory)
    # rfr_final_after_grid_search('Random Test Train Split', X_train, y_train, X_test, y_test)

    # random_forest_ada_boost(X_train, y_train, X_test, y_test, train_split)
    # gradient_boosting_regressor(X, y, X_train, y_train, X_test, y_test, train_split)
