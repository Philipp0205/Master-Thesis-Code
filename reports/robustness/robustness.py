from os import listdir
from os.path import isfile, join

import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import learner.data_preprocessing as pp
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import mean_squared_error, mean_absolute_error

import learner.data_preprocessing as dp


def missing_vt_pairings(model_data, model):
    mses = []
    df_result = pd.DataFrame(columns=['die_opening', 'mse'])

    df = pp.get_data()

    # Get all different die_openings from dataframe
    die_openings = df['die_opening'].unique()

    for die_opening in die_openings:
        # Get all rows with the die_opening
        test_rows = df.loc[df['die_opening'] == die_opening]
        # Get remaining rows
        training_rows = df.loc[df['die_opening'] != die_opening]

        # Get X and y
        X_test = test_rows[['distance', 'thickness', 'die_opening']]
        y_test = test_rows['springback']

        X_train = training_rows[['distance', 'thickness', 'die_opening']]
        y_train = training_rows['springback']

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        # Create new dataframe where with one column die opening and one column mse
        # Then sort by die opening
        df_result = df_result.concat({'die_opening': die_opening, 'mse': mse}, ignore_index=True)
        # df_result = df_result.append({'die_opening': die_opening, 'mse': mse}, ignore_index=True)

    print('-----------------')

    # Sort df by die_opening
    df_result = df_result.sort_values(by=['die_opening'])

    return df_result


def missing_values(name, model_data, model):
    X = model_data.X
    y = model_data.y

    number_of_samples = len(X)

    number_of_folds = 2

    plt.style.use(['science', 'grid'])

    mean_scores = []
    standard_errors = []
    all_folds = []
    losses = []

    # Perform Kfold cross validation with len(X) / 1 folds and increasing p every time until p =
    # len(X)
    while number_of_folds < number_of_samples - 1:
        # cv = KFold(n_splits=len(X) // number_of_folds)
        cv = KFold(number_of_folds)

        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

        # Make all scores positive
        # scores = np.abs(scores)

        folds = len(X) // number_of_folds
        standard_error = stats.sem(scores)

        print(f'folds = {number_of_folds}')
        print(f'Mean cross validation score: {scores.mean()}')

        mean_scores.append(scores.mean())
        standard_errors.append(standard_error)
        all_folds.append(number_of_folds)

        number_of_folds += 50

    # Calculate the loss for all folds
    for i in range(len(mean_scores) - 1):
        loss = mean_scores[i] - mean_scores[i + 1]
        losses.append(loss)

    # save folds scores and losses as csv file
    df = pd.DataFrame({'folds': all_folds, 'mean_rmse': mean_scores})
    path = f'{pp.root_directory()}/learner/reports/robustness/results/csv/'
    df.to_csv(f'{path}{name}.csv', index=False)

    # save mean loss as csv file
    df = pd.DataFrame({'losses': losses})
    # df.to_csv('losses.csv', index=False)

    # Get mean of all losses
    mean_loss = np.mean(losses)

    # Add mean loss as descriotion to plot
    plt.title(f'Losses {name}')

    plt.plot(all_folds, mean_scores, label=f'average loss = {round(mean_loss, 4)}')
    plt.legend()
    plt.xlabel(f'Number of folds')
    plt.ylabel('Mean RMSE')

    path = f'{pp.root_directory()}/learner/reports/robustness/results/'

    plt.savefig(f'{path}missing_values_{name}.png', dpi=600)
    plt.clf()

    return mean_loss


def calculate_variance_of_cross_validation(X, y, model):
    scores = cross_val_score(model, X, y, cv=5)

    variance_of_cv = np.var(scores, ddof=1)
    return variance_of_cv


def test_for_different_test_train_split(df, model, name):
    root = dp.root_directory()
    path = f'{root}/reports/robustness/results/csv/'

    # Read missing_values.csv file
    df_result = pd.read_csv(f'{path}missing_values.csv')

    # Delete all rows of df result where model_name is equal to name
    df_result = df_result[df_result.model_name != name]

    # iterate from 0.1 to 1 in 0.1 steps
    for i in range(1, 10):
        print(f'split = {i / 10}')
        split = pp.random_split(df, i / 10)

        model.fit(split.X_train, split.y_train)

        # Predict with the model 10 times and calculate the mean of the mse
        mses = []
        for i2 in range(20):
            y_pred = model.predict(split.X_test)
            mse = mean_squared_error(split.y_test, y_pred)
            mses.append(mse)

        mse = np.mean(mses)
        print(f'MSE: {round(mse, 3)}')
        # Concrat df_result with new row
        df_result = df_result.append({'model_name': name,
                                      'test_train_split': i / 10, 'mse': mse},
                                     ignore_index=True)

        # Save df as csv file
        df_result.to_csv(f'{path}missing_values.csv', index=False)

    # Get all unique model names from df_result
    model_names = df_result['model_name'].unique()

    plt.style.use(['science', 'grid'])

    # Iterate over all model names
    for model_name in model_names:
        # Get all rows where model_name is equal to model_name
        df_for_plot = df_result[df_result.model_name == model_name]

        # Get x and y values
        x = df_for_plot['test_train_split']
        x = x * 100

        y = df_for_plot['mse']

        plt.plot(x, y, label=model_name)

    # plt.title(f'Missing values Test')
    plt.xlabel('Percentage of data used for testing')
    plt.ylabel('MSE')
    plt.legend()

    # Place legend on the right out side plot
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # make legend transparent
    plt.legend(framealpha=0)

    plt.savefig(f'{path}missing_values_plot.png', dpi=600, transparent=True)
    plt.clf()


def test_with_noise(model_data, model):
    # Get X and y
    X = model_data.X_test
    y = model_data.y_test

    # Merge X and y in dataframe
    df = pd.concat([X, y], axis=1)

    noisy_distance = create_noise_for_feature(df, 'distance')
    noisy_thickness = create_noise_for_feature(df, 'thickness')
    noisy_die_opening = create_noise_for_feature(df, 'die_opening')
    noise_springback = create_noise_for_feature(df, 'springback')

    noise = pd.DataFrame({'distance': noisy_distance, 'thickness': noisy_thickness,
                          'die_opening': noisy_die_opening, 'springback': noise_springback})

    # Add noise to df
    # noisy_data = df.append(noise, ignore_index=True)

    # Results is a dataframe with 3 columns: noise, rmse, variance
    results = pd.DataFrame(columns=['noise', 'rmse'])

    # max_iterations = len(df) // 2
    max_iterations = 11

    # Add noise to dataframe in 1% steps until 50%
    for i in range(1, max_iterations):
        print(i)
        # i percent of the dataframe
        noise_fragment = noise.sample(frac=i / max_iterations)
        # noise_fragment = noise[:i]

        X_noise = noise_fragment[['distance', 'thickness', 'die_opening']]
        print(len(X_noise))
        y_noise = noise_fragment['springback']

        # Add noise to X_train and y_train
        X_concat = pd.concat([model_data.X_train, X_noise], ignore_index=True)
        y_concat = pd.concat([model_data.y_train, y_noise], ignore_index=True)

        # Train model
        model.fit(X_concat, y_concat)


        # Fit Predict 10 times
        rmse_list = []
        mse_list = []
        mae_list = []

        for i2 in range(1):
            y_pred = model.predict(model_data.X_test)

            mse = mean_squared_error(model_data.y_test, y_pred)
            rmse = np.sqrt(mse)

            rmse_list.append(rmse)
            mse_list.append(mse)
            mae_list.append(mean_absolute_error(model_data.y_test, y_pred))

        # Calculate mean of rmse
        rmse = np.mean(rmse_list)
        mse = np.mean(mse_list)
        mae = np.mean(mae_list)

        # Add results to dataframe with concat
        results = pd.concat(
            [results, pd.DataFrame({'noise': [i / max_iterations], 'mse': [mse], 'rmse': [rmse],
                                    'mae': [mae]
                                    })], ignore_index=True)

        # Remove noise from dataframe
        # noisy_data = noisy_data[:-len(noise_fragment)]

    # Plot results
    plt.style.use(['science', 'grid'])

    plt.plot(results['noise'], results['rmse'], label='MSE')

    plt.legend()

    # Place legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel('Noise')
    plt.ylabel('Error')
    plt.title('Error with noise')
    plt.savefig('error_with_noise1.png', dpi=600)

    average_rmse = np.mean(results['rmse'])

    return average_rmse


def create_noise_for_feature(df, feature_name):
    feature = df[feature_name]

    # Mean of feature
    mu = feature.mean()
    # Standard deviation of feature
    sigma = feature.std()

    noise = np.random.normal(mu, sigma, 300)

    return noise


def plot_results(name):
    root_dir = pp.root_directory()
    path = f'{root_dir}/reports/robustness/results/csv/'

    # ax = plt.subplot(111)

    plt.style.use(['science', 'grid', 'ieee'])

    # Get all csv file names in directory
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for file_name in files:
        # Read csv file
        df = pd.read_csv(f'{path}{file_name}')

        # Plot results
        plt.plot(df['folds'], df['mean_rmse'], label=name)

    # set x label for ax
    # ax.set_xlabel('Number of folds')
    # ax.set_ylabel('Mean RMSE')

    plt.xlabel('Number of folds')
    plt.ylabel('Mean RMSE')

    plt.legend()

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)

    plt.savefig(
        f'{root_dir}/learner/reports/robustness/results/missing_values_consolidated.png',
        dpi=600)


def variance_missing_values(model_data, model):
    root = dp.root_directory()
    path = f'{root}/reports/robustness/results/csv/'

    # Read missing_values.csv file
    df_result = pd.read_csv(f'{path}missing_values.csv')

    # Get all unique model names from df_result
    model_names = df_result['model_name'].unique()

    plt.style.use(['science', 'grid', 'ieee'])

    # Create new results with columns models name and variance
    df_variance = pd.DataFrame(columns=['model_name', 'variance'])

    # Iterate over all model names
    for model_name in model_names:
        # Calculate variance of the mse
        df = df_result[df_result['model_name'] == model_name]

        # Get mses from dataframe
        mses = df['mse'].tolist()

        # Calculate variance
        variance = np.var(mses)

        # Add results to df_variance
        df_variance = pd.concat(
            [df_variance, pd.DataFrame({'model_name': [model_name], 'variance': [variance]})],
            ignore_index=True)

    # Plot barchart
    plt.bar(df_variance['model_name'], df_variance['variance'])

    # Annotate each bar with variance
    # for index, row in df_variance.iterrows():
    #     variance = row['variance']
    #     plt.annotate(row['variance'], xy=(index, row['variance']))

    plt.xlabel('Model name')
    plt.ylabel('Variance')

    plt.savefig(f'{path}variance_missing_values.png', dpi=600, transparent=True)


def robustness_report(name, model_data, model, y_pred):
    print('------- ROBUSTNESS REPORT --------')
    # df = missing_vt_pairings(model_data, model)

    # print(df)

    # Get mses from dataframe
    # mses = df['mse'].tolist()

    # print('Average MSEs: ', m(mses) / len(mses))

    av_rmse = test_with_noise(model_data, model)
    # mean_loss = missing_values(name, model_data, model)
    # plot_results(name)

    # Merge model_data.Xand model_data.y
    df = pd.concat([model_data.X, model_data.y], axis=1)

    # test_for_different_test_train_split(df, model, name)
    # variance_missing_values(model_data, model)


    print('--------------')

    # print('Mean Loss with missing values: ', mean_loss)
    print('Average RMSE with noise: ', round(av_rmse, 3))
