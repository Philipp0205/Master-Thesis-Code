import os
import pandas as pd
import matplotlib.pyplot as plt

from zyklen.models.springback_model import SpringbackModel


def plot_all_cycles(input_directory, output_directory):
    file_names = os.listdir(input_directory)
    split_values = [0, 10, 10, 20, 20, 30, 30, 40]

    springbacks = []

    for name in file_names:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.grid(axis='y')

        df = pd.read_csv(f'{input_directory}{name}', delimiter=';')

        for x, y in pairwise(split_values):
            # Get all values where 'Prüfzeit' is between the split values
            df_split = df.loc[(df['Prüfzeit'] > x) & (df['Prüfzeit'] < y)]

            # Get point where the force is max
            # max_force = df_split['Standardkraft'].max()
            # max_force_row = df_split.loc[df_split['Standardkraft'] == max_force]

            max_force = df_split['Standardkraft'].max()
            max_force_row = df_split.loc[df_split['Standardkraft'] == max_force]

            max_distance= df_split['Standardweg'].max()
            max_distance_rows = df_split.loc[round(df_split['Standardweg'], 2) == round(max_distance, 2)]

            # Get last row of max_distance_rows
            last_row = max_distance_rows.iloc[-1]
            last_row_x = last_row['Prüfzeit']
            last_row_y = last_row['Standardkraft']
            last_row_distance = last_row['Standardweg']


            # Get point where the force is below 1 and after the max force
            df_after_max = df_split[df_split['Prüfzeit'] > max_force_row['Prüfzeit'].values[0]]
            min_force_row = df_after_max[df_after_max['Standardkraft'] < 1].iloc[0]
            min_force = min_force_row['Standardkraft']
            min_force_distance = min_force_row['Standardweg']

            # springback = max_force_row['Standardweg'].values[0] - min_force_row['Standardweg']
            springback = last_row_distance - min_force_distance

            max_x = max_force_row['Prüfzeit']
            min_x = min_force_row['Prüfzeit']

            max_y = max_force_row['Standardkraft']
            min_y = min_force_row['Standardkraft']

            x_axis = df_split['Prüfzeit']
            y_axis = df_split['Standardkraft']

            # min_x_distance = max_distance_row['Prüfzeit']
            # min_y_distance = max_distance_row['Standardkraft']

            max_distance = max_force_row['Standardweg']
            min_distance = min_force_row['Standardweg']

            springbacks.append(SpringbackModel(name, last_row_distance, springback))

            # ax1.scatter(max_x, max_y, color='red', marker='x')
            ax1.scatter(min_x, min_y, color='red', marker='x')
            ax1.scatter(last_row_x, last_row_y, color='green', marker='x')
            # ax1.scatter(min_x_distance, min_y_distance, color='green', marker='x')

            # ax1.annotate(f'{round(max_force, 2)} N, \n{round(max_distance.iat[0], 2)} mm', (max_x.iat[0], max_y.iat[0]), fontsize=6)
            ax1.annotate(f'{round(min_force, 2)} N, \n{round(min_distance, 2)} mm', (min_x, min_y), fontsize=6)
            ax1.annotate(f'{round(last_row_distance, 2)} mm', (last_row_x, last_row_y), fontsize=6)
            ax1.plot(x_axis, y_axis, label=f'{round(springback, 3)}')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Standardkraft')
            ax1.legend()

        x2_axis = df['Prüfzeit']
        y2_axis = df['Standardweg']
        ax2.set_ylabel('Distance')
        ax2.plot(x2_axis, y2_axis, label='Distance', color='grey')
        ax1.legend()

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

        # Put a legend below current axis
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
                   fancybox=True, shadow=True, ncol=5)
        plt.savefig(f'{output_directory}{name}{x}.jpg', dpi=600)


    fig, ax1 = plt.subplots()

    # Get all distances of springback_models
    x_distances = [model.distance for model in springbacks]
    # Get all springbacks of springback_models
    y_springbacks = [model.springback for model in springbacks]

    ax1.scatter(x_distances, y_springbacks, color='red',)
    ax1.set_xlabel('Distance [mm]')
    ax1.set_ylabel('Springback [mm]')
    ax1.set_title('Springback of all springback models')
    ax1.grid(True)

    plt.savefig(f'{output_directory}all-springbacks')


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)
