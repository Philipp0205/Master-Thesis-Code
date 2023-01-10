import pickle
import time
from pathlib import Path

import numpy as np

from learner.data_preprocessing import *
# import mean_squared_error from sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

import pandas as pd


# import leave one out cross validation
# Entry function for the missing values test.
def missing_values_main(X_train, y_train, X_test, y_test):
    # Merge X_train and y_train dataframes
    train = pd.concat([X_train, y_train], axis=1)

    losses = []

    for i in range(0, 10):
        remaining_X_test, remaining_y_test, removed_X_test, removed_Y_test \
            = missing_vt_combinations_test(train, 4)

        # Train model with missing values
        model_with_missing_values = train_tuned_random_forest(remaining_X_test, remaining_y_test)
        mae, mse, rmse, r2 = test_tuned_random_forest(model_with_missing_values, X_test, y_test)

        # Train model with all data
        model_without_missing_values = train_tuned_random_forest(X_train, y_train)
        mae_without_missing_values, mse, rmse, r2 = test_tuned_random_forest(model_without_missing_values, X_test,
                                                                             y_test)
        # Loss off accuracy
        loss = mae_without_missing_values - mae
        losses.append(loss)

    # Mean loss of accuracy
    mean_loss = np.mean(losses)
    print(f'Mean loss of accuracy: {mean_loss}')


def missing_values_main_2(df, number_of_groups):
    print('------- Missing values 2 test --------')

    # Define grouping features
    groups = df['die_opening']

    # Create group k fold
    gkf = GroupKFold(n_splits=number_of_groups)

    # Get the data
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # pipe = Pipeline(
    #     [("scaler", MinMaxScaler()), ("rfr", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
    #                                                                min_samples_split=4, n_estimators=10))])

    model = RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                  min_samples_split=4, n_estimators=10)

    cv = GroupKFold(n_splits=number_of_groups)

    # scores = cross_val_score(rfr, X, y=y, groups=groups, cv=GroupKFold(n_splits=3))
    scores = cross_val_score(model, X, y, cv=cv, groups=groups)

    print(f'Cross validation scores: {scores}')

    # for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #
    #     model = train_tuned_random_forest(X_train, y_train)
    #     mae, mse, rmse, r2 = test_tuned_random_forest(model, X_test, y_test)


def train_tuned_random_forest(X_train, y_train):
    print('------- Train tuned random forest --------')
    print(f'Number of training samples: {len(X_train)}')
    print(f'Number of testing samples: {len(X_test)}')
    print('-----')

    pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("rfr", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                                                   min_samples_split=4, n_estimators=10))])
    sample = X_test.iloc[0]
    start = time.time()

    pipe.fit(X_train.to_numpy(), y_train)
    stop = time.time()
    s = stop - start
    print(f"Training time: {s} s")

    return pipe


def test_tuned_random_forest(model, X_test, y_test):
    print('------- Testing tuned random forest --------')
    sample = X_test.iloc[0]
    start = time.time()
    model.predict([sample])
    stop = time.time()
    ms = (stop - start) * 1000

    print(f"Prediction time (runtime): {ms} ms")
    save_trained_model(model, 'tuned_random_forest')

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2


# Save trained model to disk
def save_trained_model(model, name):
    project_root = get_project_root()
    output_directory = project_root / 'learner' / 'rfr' / 'saved_models'

    with open(f'{output_directory}/{name}.model', 'wb') as f:
        pickle.dump(model, f)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    # Constants
    train_split = 30
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    # Load data
    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')

    # X, y, X_train, y_train, X_test, y_test = non_random_split(df, train_split)
    # missing_values_main(X_train, y_train, X_test, y_test)

    missing_values_main_2(df, 5)

    print('Done!')
