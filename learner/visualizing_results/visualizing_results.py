import os

import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

import learner.data_preprocessing as dp
import matplotlib.pyplot as plt
import scienceplots
import learner.support_vector_machine.support_vector_machine as svm
import learner.random_forest.random_forest as rfr
import learner.extra_trees.extra_trees as etr
import learner.linear_regression.linear_regression as lr
import learner.mlp.multi_layer_perceptron as mlp
import learner.data_preprocessing as dp

import learner.data_preprocessing as dp


def plot_all_data(df):
    result_df = pd.DataFrame(
        columns=['die_opening', 'distance', 'thickness', 'mean_springback'])

    plt.style.use(['science', 'ieee'])

    # Get all die openings from df
    die_openings = df['die_opening'].unique()

    mean_df = pd.DataFrame(columns=['die_opening', 'distance', 'springback'])

    # Iterate over all die openings
    for die_opening in die_openings:

        thicknesses = df[df['die_opening'] == die_opening]['thickness'].values

        # Iterate over all thicknesses
        for thickness in thicknesses:
            # get distances
            x = df[(df['die_opening'] == die_opening) & (df['thickness'] == thickness)][
                'distance'].values
            # get springbacks
            y = df[(df['die_opening'] == die_opening) & (df['thickness'] == thickness)][
                'springback'].values

            # Get all distances for die_opening
            distances = df[df['die_opening'] == die_opening]['distance'].unique()

            distances2 = []
            springbacks2 = []

            # Iterate over all distances
            for distance in distances:
                # Get mean springback for each distance
                springbacks = df[(df['die_opening'] == die_opening) & (
                        df['thickness'] == thickness) & (df['distance'] == distance)][
                    'springback'].values

                if (len(springbacks) == 0):
                    continue

                mean_sb = np.mean(springbacks)

                distances2.append(distance)
                springbacks2.append(mean_sb)

            # Plot the data
            # plt.plot(x, y, 'o', label=f'Die opening: {die_opening}', markersize=1,
            #          linestyle='None')

            # Dataframe from distances 2 and springbacks2
            df2 = pd.DataFrame({'die_opening': die_opening, 'distance': distances2,
                                'mean_springback': springbacks2, 'thickness': thickness})
            # Sort by distance
            # df2 = df2.sort_values(by=['distance'])

            # Concat result_df and df2
            result_df = pd.concat([result_df, df2], ignore_index=True)

            # Sort result_df by die opening
            result_df = result_df.sort_values(by=['die_opening'])

            # Save result_df to csv
            result_df.to_csv('results/result_df.csv', index=False)

    return result_df


def visualize_mean_spring_backs(df):
    plt.style.use(['science', 'ieee', 'grid'])
    # for each die opening
    for die_opening in df['die_opening'].unique():
        # for each thickness and die opening
        for thickness in df[df['die_opening'] == die_opening]['thickness'].unique():
            # Get all data where die opening and and thicknesses are equal to the current
            # die opening and thickness
            df2 = df[(df['die_opening'] == die_opening) & (df['thickness'] == thickness)]

            # Sort by distance
            df2 = df2.sort_values(by=['distance'])

            # Plot the data
            plt.plot(df2['distance'], df2['mean_springback'], 'o',
                     label=f'Die opening: {die_opening}',
                     markersize=1,
                     linestyle='solid')

            plt.xlabel('Distance [mm]')
            plt.ylabel('Springback [mm]')
            plt.savefig(f'results/plot_{die_opening}_{thickness}.png', dpi=300)
            plt.clf()


def visualize_model_vs_example(model, df, die_opening, thickness):
    plt.style.use(['science', 'grid', 'ieee'])

    # Get all data where die opening and thicknesses are equal to the current
    df_part = df.loc[(df['die_opening'] == die_opening) & (df['thickness'] == thickness)]

    # Sort df_test by distance
    df_part = df_part.sort_values(by=['distance'])

    X_test = df_part.drop(['springback'], axis=1)
    y_test = df_part['springback']

    data = dp.non_random_split(df, 30)

    #  Merge data.X_train and data.y_train
    df_train = pd.concat([data.X_train, data.y_train], axis=1)

    # Remove all rows where die_opening is equal to die_opening and
    # thickness is equal to thickness
    # This is done to prevent overfitting
    # -> No bias
    df_train_new = df_train.loc[~((df_train['die_opening'] == die_opening) & (
            df_train['thickness'] == thickness))]

    X_train2 = df_train_new.drop(['springback'], axis=1)
    y_train2 = df_train_new['springback']

    model.fit(X_train2, y_train2)

    y_pred = model.predict(X_test)

    # Merge y_pred and df_test
    df_part['pred_springback'] = y_pred

    # Create new dataframe with one column distance and one mean_springback
    df_mean = pd.DataFrame(
        columns=['distance', 'mean_springback', 'pred_springback'])

    # Iterate over all distances in df_test
    for distance in df_part['distance'].unique():
        # Get all springbacks for distance
        springbacks = df_part.loc[df_part['distance'] == distance]['springback']
        # Calculate mean springback
        mean_springback = springbacks.mean()

        pred_springback = df_part.loc[df_part['distance'] == distance][
            'pred_springback'].mean()

        # Concat distance and mean_springback to df_mean
        df_mean = pd.concat(
            [df_mean, pd.DataFrame([[distance, mean_springback, pred_springback]],
                                   columns=['distance',
                                            'mean_springback', 'pred_springback'])])

    return df_mean


def visualize_all_results_for_example(die_opening, thickness):
    df = dp.get_data()
    md = dp.non_random_split(df, 30)
    md2 = dp.random_split(df, 0.2)

    svm_model, y_pred = svm.svr(df, md)
    df_result_svm = visualize_model_vs_example(svm_model, df, die_opening, thickness)

    rf_model, y_pred = rfr.random_forest(md)
    df_result_rf = visualize_model_vs_example(rf_model, df, die_opening, thickness)

    et_model, y_pred = etr.extra_trees(md)
    df_result_et = visualize_model_vs_example(et_model, df, die_opening, thickness)

    mlp_model, y_pred = mlp.mlp(md)
    df_result_mlp = visualize_model_vs_example(mlp_model, df, die_opening, thickness)

    plt.plot(df_result_svm['distance'], df_result_svm['mean_springback'], label='Target',
             marker='o',
             color='black',
             markersize=2,
             linestyle='solid')

    # Calculate RMSE for SVM
    rmse_svm = np.sqrt(mean_squared_error(df_result_svm['mean_springback'],
                                          df_result_svm['pred_springback']))

    plt.plot(df_result_svm['distance'], df_result_svm['pred_springback'], label=f'SVM ({rmse_svm:.2f})',
             marker='o',
             markersize=2,
             color='red',
             linestyle='dotted')

    # Calculate RMSE for RF
    rmse_rf = np.sqrt(mean_squared_error(df_result_rf['mean_springback'],
                                         df_result_rf['pred_springback']))

    plt.plot(df_result_rf['distance'], df_result_rf['pred_springback'], label=f'RF ({rmse_rf:.2f})',
             marker='o',
             markersize=2,
             color='blue',
             linestyle='dotted')
    # Calculate RMSE for ET
    rmse_et = np.sqrt(mean_squared_error(df_result_et['mean_springback'],
                                         df_result_et['pred_springback']))

    plt.plot(df_result_et['distance'], df_result_et['pred_springback'], label=f'ET ({rmse_et:.2f})',
             marker='o',
             markersize=2,
             color='green',
             linestyle='dotted')

    # Calculate RMSE for MLP
    rmse_mlp = np.sqrt(mean_squared_error(df_result_mlp['mean_springback'],
                                          df_result_mlp['pred_springback']))

    plt.plot(df_result_mlp['distance'], df_result_mlp['pred_springback'], label=f'MLP ({rmse_mlp:.2f})',
             marker='o',
             markersize=2,
             color='purple',
             linestyle='dotted')

    plt.legend()
    # Make legend transparent
    plt.gca().get_legend().get_frame().set_alpha(0)

    plt.xlabel('$y_p$')
    plt.ylabel('$SB$')
    plt.savefig(f'performance_{die_opening}_{thickness}.png', dpi=600, transparent=True)

    plt.clf()


if __name__ == '__main__':
    print('------- Visualizing results --------')
    # Load data
    df = dp.get_data()
    df2 = plot_all_data(df)

    # \item V20 t3 (6.6) (for now)
    # \item V30, t1.5 (15)
    # \item V50 t0.5 (100)

    visualize_all_results_for_example(20, 3.0)
    visualize_all_results_for_example(20, 2.0)

    visualize_all_results_for_example(30, 1.5)
    visualize_all_results_for_example(20, 1.5)

    visualize_all_results_for_example(50, 0.5)
    visualize_all_results_for_example(40, 0.5)
