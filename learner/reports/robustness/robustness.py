from os import listdir
from os.path import isfile, join

import numpy as np
from joblib.numpy_pickle_utils import xrange
from sklearn.model_selection import KFold, cross_val_score

import learner.data_preprocessing as preprocessing
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import mean_squared_error, mean_absolute_error


def missing_vt_pairings(model_data, model):
    mses = []
    df_result = pd.DataFrame(columns=['die_opening', 'mse'])

    df = preprocessing.get_data()

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

    # Perform Kfold cross validation with len(X) / 1 folds and increasing p every time until p = len(X)
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
    path = f'{preprocessing.root_directory()}/learner/reports/robustness/results/csv/'
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

    path = f'{preprocessing.root_directory()}/learner/reports/robustness/results/'

    plt.savefig(f'{path}missing_values_{name}.png', dpi=600)
    plt.clf()

    return mean_loss


def calculate_variance_of_cross_validation(X, y, model):
    scores = cross_val_score(model, X, y, cv=5)

    variance_of_cv = np.var(scores, ddof=1)
    return variance_of_cv


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

    max_iterations = len(df) // 2

    # Add noise to dataframe in 1% steps until 50%

    for i in range(1, max_iterations, 5):
        # Get i rows from noise
        noise_fragment = noise.iloc[:i]

        X_noise = noise_fragment[['distance', 'thickness', 'die_opening']]
        y_noise = noise_fragment['springback']

        # Add noise to X_train and y_train
        X_concat = pd.concat([model_data.X_train, X_noise], ignore_index=True)
        y_concat = pd.concat([model_data.y_train, y_noise], ignore_index=True)

        # Train model
        model.fit(X_concat, y_concat)

        # Predict
        y_pred = model.predict(model_data.X_test)

        mse = mean_squared_error(model_data.y_test, y_pred)
        rmse = np.sqrt(mse)

        # Add results to dataframe with concat
        results = pd.concat(
            [results, pd.DataFrame({'noise': [i / len(df)], 'mse': [mse], 'rmse': [rmse],
                                    })], ignore_index=True)

        # Remove noise from dataframe
        # noisy_data = noisy_data[:-len(noise_fragment)]

    rmses = results['rmse']
    averae_rmse = np.mean(rmses)

    # Plot results
    plt.style.use(['science', 'grid'])

    # plt.plot(results['noise'], results['rmse'], label='RMSE')

    plt.legend()

    # Place legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel('Noise')
    plt.ylabel('Error')
    plt.title('Error with noise')
    plt.savefig('error_with_noise1.png', dpi=600)

    return averae_rmse


def create_noise_for_feature(df, feature_name):
    feature = df[feature_name]

    # Mean of feature
    mu = feature.mean()
    # Standard deviation of feature
    sigma = feature.std()

    noise = np.random.normal(mu, sigma, feature.shape)

    return noise


def plot_results(name):
    root_dir = preprocessing.root_directory()
    path = f'{root_dir}/learner/reports/robustness/results/csv/'

    # ax = plt.subplot(111)

    plt.style.use(['science', 'grid'])

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


def robustness_report(name, model_data, model, y_pred):
    print('------- ROBUSTNESS REPORT --------')
    # df = missing_vt_pairings(model_data, model)

    # print(df)

    # Get mses from dataframe
    # mses = df['mse'].tolist()

    # print('Average MSEs: ', sum(mses) / len(mses))

    av_rmse = test_with_noise(model_data, model)
    mean_loss = missing_values(name, model_data, model)
    plot_results(name)

    print('--------------')

    print('Mean Loss with missing values: ', mean_loss)
    print('Average RMSE with noise: ', av_rmse)

    print('\n------- END ROBUSTNESS REPORT --------\n')
