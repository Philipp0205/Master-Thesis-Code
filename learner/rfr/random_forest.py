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

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, LeavePOut, LeaveOneOut, KFold

import matplotlib.pyplot as plt
import scienceplots
from scipy import interpolate

import pandas as pd


# import leave one out cross validation
# Entry function for the missing values test.
def missing_values_main_2(df, number_of_groups):
    print('------- Missing values 2 test --------')

    # Define grouping features
    groups = df['die_opening']

    # Create group k fold
    gkf = GroupKFold(n_splits=number_of_groups)

    # Get the data
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    model = RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                  min_samples_split=4, n_estimators=10)

    groups = [10, 20, 30, 40, 50]

    cv = LeaveOneGroupOut()

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={groups[train_index]}")
        print(f"  Test:  index={test_index}, group={groups[test_index]}")

    scores = cross_val_score(model, X, y, cv=cv, groups=groups)

    # cv = GroupKFold(n_splits=number_of_groups)
    # scores = cross_val_score(rfr, X, y=y, groups=groups, cv=GroupKFold(n_splits=3))

    print(f'Cross validation scores: {scores}')


def group_k_fold_test(df):
    plt.figure(figsize=(10, 2))
    plt.title("GroupKFold")

    axes = plt.gca()
    axes.set_frame_on(False)

    print('------- Group k fold test --------')
    # Define grouping features
    groups = df['die_opening']

    # Create group k fold
    gkf = GroupKFold(n_splits=5)
    logo = LeaveOneGroupOut()

    # Get the data
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # model = Pipeline(
    #     [("scaler", MinMaxScaler()), ("rfr", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
    #                                                                min_samples_split=4, n_estimators=10))])

    model = RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                  max_depth=30, min_samples_leaf=2,
                                  min_samples_split=4, n_estimators=10)

    # Calculate scores
    # q: How do I add the r2 to the scoring method in cross_val_score?
    scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring='r2')
    print(f'Cross validation scores: {scores}')
    # Caluclate mean r2 score
    mean_r2 = np.mean(scores)
    print(f'Mean r2 score: {mean_r2}')

    scores2 = cross_val_score(model, X, y)

    print('---')

    print(f'Cross validation scores: {scores2}')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
        train_groups = groups[train_index].unique()
        test_groups = groups[test_index].unique()
        print(f"Fold {i}:")
        print(f" train: {train_groups}")
        print(f" test: {test_groups}")
        print('------------------------------')


def train_tuned_random_forest(X_train, y_train, X_test, y_test):
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


def get_model():
    pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("rfr", RandomForestRegressor(bootstrap=True, criterion='absolute_error',
                                                                   min_samples_split=4, n_estimators=10))])

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


def k_fold_CV(df, model):
    print('------- Leave p out CV test --------')

    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    number_of_samples = len(X)

    number_of_folds = 2

    plt.style.use(['science', 'grid'])

    mean_scores = []
    all_folds = []
    losses = []

    # Perform Kfold cross validation with len(X) / 1 folds and increasing p every time until p = len(X)
    while number_of_folds < number_of_samples - 1:
        # cv = KFold(n_splits=len(X) // number_of_folds)
        cv = KFold(number_of_folds)

        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        folds = len(X) // number_of_folds
        print(f'folds = {number_of_folds}')
        print(f'Mean cross validation score: {scores.mean()}')

        # Make all scores positive
        scores = np.abs(scores)

        mean_scores.append(scores.mean())
        all_folds.append(number_of_folds)

        number_of_folds += 10

    # Calculate the loss for all folds
    for i in range(len(mean_scores) - 1):
        loss = mean_scores[i] - mean_scores[i + 1]
        losses.append(loss)

    # save folds scores and losses as csv file
    df = pd.DataFrame({'folds': all_folds, 'scores': mean_scores})
    df.to_csv('folds_scores.csv', index=False)

    # save mean loss as csv file
    df = pd.DataFrame({'losses': losses})
    df.to_csv('losses.csv', index=False)


    # Get mean of all losses
    mean_loss = np.mean(losses)

    # Add mean loss as descriotion to plot
    plt.title(f'Losses for different number of folds')

    plt.plot(all_folds, mean_scores, label=f'average loss = {round(mean_loss, 4)}')
    plt.legend()
    plt.xlabel('Number of folds')
    plt.ylabel('Mean RMSE')

    plt.savefig('kfold.png', dpi=600)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    # Constants
    train_split = 30
    project_root = get_project_root()
    input_directory = project_root / 'data' / 'dataset'

    # Load data
    df = pd.read_csv(input_directory / 'consolidated.csv', delimiter=',')

    model = get_model()

    k_fold_CV(df, model)

    print('Done!')
