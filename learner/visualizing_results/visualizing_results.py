import numpy as np
import pandas as pd

import learner.data_preprocessing as dp
import matplotlib.pyplot as plt
import scienceplots
import learner.support_vector_machine.support_vector_machine as svm
import learner.random_forest.random_forest as rfr
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

            # plt.plot(df2['distance'], df2['springback'], 'o',
            #          label=f'Die opening: {die_opening}',
            #          markersize=1,
            #          linestyle='solid')
            #
            # plt.xlabel('Distance [mm]')
            # plt.ylabel('Springback [mm]')
            # plt.savefig(f'results/plot_{die_opening}_{thickness}.png', dpi=300)
            # plt.clf()

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

    df_part = df.loc[(df['die_opening'] == die_opening) & (df['thickness'] == thickness)]

    # Sort df_test by distance
    df_part = df_part.sort_values(by=['distance'])

    X_test = df_part.drop(['springback'], axis=1)
    y_test = df_part['springback']

    data = dp.non_random_split(df_part, 30)
    model.fit(data.X_train, data.y_train)

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

    print()

    # # Sort df_mean by distance
    # df_mean = df_mean.sort_values(by=['distance'])
    #
    # # Concat X_test and y_pred to df_test
    # df_part = pd.concat([df_part, pd.DataFrame(y_pred, columns=['y_pred'])], axis=1)
    #
    # # Sort df_test by distance
    # df_part = df_part.sort_values(by=['distance'])
    #
    # # Plot y_test and y_pred
    # # plt.plot(X_test['distance'], y_test, label='y_test', marker='o', markersize=2,
    # #          linestyle='None')
    #
    # # Create dataframe with one column distance and one mean_springback
    # df_result = pd.DataFrame(columns=['distance', 'mean_springback', 'pred_springback'])
    #
    # # Concat df_mean to df_result
    # df_result = pd.concat([df_result, df_mean], ignore_index=True)
    #
    # # Concat df_test to df_result where distance is equal
    # df_result = pd.concat([df_result, df_part[['distance', 'y_pred']]], ignore_index=True)

    # plt.plot(df_mean['distance'], df_mean['mean_springback'], label='Target',
    # marker='o',
    #          markersize=2,
    #          linestyle='solid')
    # plt.plot(X_test['distance'], y_pred, label='SVM', marker='o', markersize=2,
    #          linestyle='dotted')
    # plt.legend()
    # plt.savefig('y_test.png', dpi=600)

    return df_mean


def visualize_all_results():
    df = dp.get_data()

    md = dp.non_random_split(df, 30)

    svm_model, y_pred = svm.svr(df, md.X_train, md.y_train, md.X_test, md.y_test)
    df_result_svm = visualize_model_vs_example(svm_model, df, 20, 3)

    rf_model, y_pred = rfr.random_forest(df, md.X_train, md.y_train, md.X_test, md.y_test)
    df_result_rf = visualize_model_vs_example(rf_model, df, 20, 3)

    plt.plot(df_result_svm['distance'], df_result_svm['mean_springback'], label='Target',
                marker='o',
                markersize=2,
                linestyle='solid')
    plt.plot(df_result_svm['distance'], df_result_svm['pred_springback'], label='SVM', marker='o',
                markersize=2,
                linestyle='dotted')
    plt.plot(df_result_rf['distance'], df_result_rf['pred_springback'], label='RF', marker='o',
                markersize=2,
                linestyle='dotted')

    plt.legend()
    plt.savefig('performance_svm_rf.png', dpi=600)




if __name__ == '__main__':
    print('------- Visualizing results --------')
    # Load data
    df = dp.get_data()
    df2 = plot_all_data(df)

    visualize_all_results()

