import random
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split

from learner.model_data import ModelDataModel
from misc import folders


# Split the data into a training and testing set.
# The functions split the data very specific like described in the thesis.
# Out of the parameters range (V10-50 and t1-3) one V-set is removed and used as
# testing set.
# The remaining V-sets are used as training set, so it is a non-random split.
#
# The train_split value defined in the thesis is 30.
# This means that all rows with a die opening of 30mm are used as testing set.
#
# The goal of this split to test if the model generalizes well on new data, because it
# has never seen
# data with dat V (die opening) yet.
# The model should be able to predict the springback for this V.
#
# :param df: The dataframe to split
# :param train_split: The die_opening which should be excluded from the training set
def non_random_split(df, train_split):
    # correlations(df)

    print(f'Number of samples: {len(df.index)}')

    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # Choose as testing set all values with an die opening of 20mm
    X_test = X.loc[X['die_opening'] == train_split]
    y_test = y.loc[X['die_opening'] == train_split]

    X_train = X[X['die_opening'] != train_split]
    y_train = y[X['die_opening'] != train_split]

    return ModelDataModel(X, y, X_train, y_train, X_test, y_test, None, None)


# Measure the robustness of the model by removing a certain number of rows in the
# training dataset.
# The test dataset stays the same.
#
# Thesis: Certain Vt combinations will get removed randomly.
#
# The model will be trained on the remaining data and the results will be compared to
# the test dataset.
# number_of_missing_values: Number of Vt combinations which will be removed from the
# training dataset.
def missing_vt_combinations_test(df, number_of_missing_values):
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
    random_numbers = random.sample(range(0, number_of_combinations),
                                   number_of_missing_values)

    # Get the rows where the index are the random numbers
    rows_to_remove = []
    for i in random_numbers:
        rows_to_remove.append(thickness_die_opening_combinations[i])

    # Remove rows from dataframe
    remaining_data = df[
        ~df[['thickness', 'die_opening']].apply(tuple, 1).isin(rows_to_remove)]

    # Get removed rows
    removed_data = df[
        df[['thickness', 'die_opening']].apply(tuple, 1).isin(rows_to_remove)]

    # print combinations which got removed
    for i in range(0, len(random_numbers)):
        print(f'Removed: {thickness_die_opening_combinations[i]}')

    remaining_X_train = remaining_data[['distance', 'thickness', 'die_opening']]
    remaining_y_train = remaining_data['springback']

    removed_X_train = removed_data[['distance', 'thickness', 'die_opening']]
    removed_y_train = removed_data['springback']

    return remaining_X_train, remaining_y_train, removed_X_train, removed_y_train


def missing_random_values_test(df, number_of_missing_values):
    print('------- Missing values: Removing random rows --------')
    # Remove random rows
    missing_data = df.sample(n=number_of_missing_values)
    remaining_data = df.drop(missing_data.index)

    return missing_data, remaining_data


def group_k_fold_CV_test(df, model, number_of_groups):
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    print('------- Group K-Fold CV --------')
    # Get the groups
    groups = df['die_opening']

    group_kfold = GroupKFold(n_splits=number_of_groups)

    # Get the data
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Visualize cross validation behaviour


# Plot the correlation matrix
def correlations(df):
    print('Plotting Correlations')
    # Create correlation matrix
    df.corr().style.background_gradient(cmap='coolwarm')

    # Plot heatmap
    plt.figure(dpi=600)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('results/correlation_matrix.png')
    plt.clf()


def get_data():
    data_folder = folders.data_directory()

    df = pd.read_csv(data_folder / 'consolidated.csv', delimiter=',')
    # Round distance of df
    df['distance'] = df['distance'].round(1)

    return df


def non_random_split_with_validation(df, train_split):
    model_data = non_random_split(df, train_split)

    # Split the training data into training and validation set
    # 0.25 x 0.8 = 0.2
    X_train, X_val, y_train, y_val = train_test_split(model_data.X_train,
                                                      model_data.y_train, test_size=0.25,
                                                      random_state=1)

    return ModelDataModel(model_data.X, model_data.y, X_train, y_train, model_data.X_test,
                          model_data.y_test, X_val, y_val)


def root_directory():
    return Path(__file__).parent.parent


def random_split(df):
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    return ModelDataModel(X, y, X_train, y_train, X_test, y_test, None, None)
