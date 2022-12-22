import os
import statistics

from matplotlib.legend import Shadow
from pandas.core.arrays.sparse import dtype
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from zyklen.archive.springback_value import SpringbackValue


def read_excel():
    df = pd.read_csv('../data/cycle_vs_single_test/cycle_measurements/4-zyklen-vergleich-4.csv', delimiter=";")

    create_force_time_chart(df)


def create_force_time_chart(df):
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Delete the second row from dataframe (it contains the units)
    df = df.drop(df.index[0])
    df = df.apply(pd.to_numeric)

    # Delete rows from dataframe where 'Prüfzeit' is 0
    # df = df[df['Prüfzeit'].astype(float) != 0]

    split_values = [10, 20, 30, 40]
    df_iterate = df

    for value in split_values:
        df_split = df_iterate[df_iterate['Prüfzeit'] <= value]
        # remove df_split from df_iterate
        df_iterate = df_iterate[df_iterate['Prüfzeit'] > value]

        x = df_split['Prüfzeit']
        y = df_split['Standardkraftsensor']

        # Get max point with y max value
        max_point = df_split.loc[df_split['Standardkraftsensor'].idxmax()]

        # Point after the max_pint where the 'Standardkraftsensor' is below 0.
        min_point = \
            df_split.loc[(df_split['Standardkraftsensor'] < 1) & (df_split['Prüfzeit'] > max_point['Prüfzeit'])].iloc[0]

        springback = max_point['Standardwegsensor'] - min_point['Standardwegsensor']

        # Scatter max_point
        ax.scatter(max_point['Prüfzeit'], max_point['Standardkraftsensor'], color='red', marker='o', s=20)
        ax.scatter(min_point['Prüfzeit'], min_point['Standardkraftsensor'], color='red', marker='o', s=20)
        # plt.text(max_point['Prüfzeit'], max_point['Standardkraftsensor'], round(springback, 2), fontsize=10)

        ax.plot(x, y, label=f'{round(springback, 2)}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Force')

    x2 = df['Prüfzeit']
    y2 = df['Standardweg']
    ax2 = ax.twinx()
    ax2.plot(x2, y2, color='grey')
    ax2.set_ylabel('Standardweg')
    ax.legend()
    plt.grid(True)
    plt.savefig('4-zyklen-vergleich-4.png')


def calculate_springback(csv_name):
    df = pd.read_csv(f'data/cycle_vs_single_test/cycle_measurements/{csv_name}',
                     delimiter=";", decimal=",")


    # Delete the second row from dataframe (it contains the units)
    df = df.drop(df.index[0])
    df = df.apply(pd.to_numeric)
    # Delete all rows where 'Prüfzeit' is below 0
    df = df[df['Prüfzeit'] >= 0]

    # Delete rows from dataframe where 'Prüfzeit' is 0
    # df = df[df['Prüfzeit'].astype(float) != 0]

    split_values = [10, 20, 30, 40]
    df_iterate = df

    springbacks = []

    for value in split_values:
        df_split = df_iterate[df_iterate['Prüfzeit'] <= value]
        # remove df_split from df_iterate
        df_iterate = df_iterate[df_iterate['Prüfzeit'] > value]

        x = df_split['Prüfzeit']
        y = df_split['Standardkraftsensor']

        # Get max point with y max value
        max_point = df_split.loc[df_split['Standardkraftsensor'].idxmax()]

        # Point after the max_pint where the 'Standardkraftsensor' is below 0.
        min_point = \
            df_split.loc[(df_split['Standardkraftsensor'] < 1) & (df_split['Prüfzeit'] > max_point['Prüfzeit'])].iloc[0]

        max_distance = df_split.loc[df_split['Standardweg'].idxmax()]

        springback = max_point['Standardwegsensor'] - min_point['Standardwegsensor']
        distance = max_distance['Standardweg']

        springbacks.append((springback, distance))

    return springbacks


def plot_all_springbacks():
    fig, ax = plt.subplots(figsize=(10, 6))

    csv_names = [
        '4-zyklen-vergleich-1.csv',
        '4-zyklen-vergleich-2.csv',
        '4-zyklen-vergleich-3.csv',
        '4-zyklen-vergleich-4.csv',
        '4-zyklen-vergleich-5.csv',
        '4-zyklen-vergleich-6.csv',
        '4-zyklen-vergleich-7.csv',
    ]

    y_values = []
    x_values = []

    data_objects = []

    all_springback_distances = []

    for name in csv_names:
        springbacks_distances = calculate_springback(name)

        for value in springbacks_distances:
            data_objects.append(SpringbackValue(value[0], round(value[1],1)))

    for point in data_objects:
        plt.scatter(point.distance, point.springback, color='blue', marker='o', s=20)

    # Get data objects where the distance is 5
    data_objects_5 = [x for x in data_objects if x.distance == 5]
    data_objects6 = [x for x in data_objects if x.distance == 6]
    data_objects7 = [x for x in data_objects if x.distance == 7]
    data_objects8 = [x for x in data_objects if x.distance == 8]

    # Get the mean of the springback values
    mean_5 = statistics.mean([x.springback for x in data_objects_5])
    mean_6 = statistics.mean([x.springback for x in data_objects6])
    mean_7 = statistics.mean([x.springback for x in data_objects7])
    mean_8 = statistics.mean([x.springback for x in data_objects8])

    # Scatter all the mean values
    plt.scatter(5, mean_5, color='red', marker='o', s=50)
    plt.scatter(6, mean_6, color='red', marker='o', s=50)
    plt.scatter(7, mean_7, color='red', marker='o', s=50)
    plt.scatter(8, mean_8, color='red', marker='o', s=50)

    # Label all mean valuees with the mean value
    plt.text(5, mean_5, round(mean_5, 2), fontsize=10)
    plt.text(6, mean_6, round(mean_6, 2), fontsize=10)
    plt.text(7, mean_7, round(mean_7, 2), fontsize=10)
    plt.text(8, mean_8, round(mean_8, 2), fontsize=10)

    plt.grid(True)
    plt.savefig('all_springbacks2.png')


def plot_the_single_bend():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all file names from directory
    file_names = os.listdir('../data/cycle_vs_single_test/single_measurements/cleared')

    y_values = []
    x_values = []

    for file_name in file_names:
        df = pd.read_csv(f'data/cycle_vs_single_test/single_measurements/cleared/{file_name}', delimiter=";")
        max_point = df.loc[df['Standardkraftsensor'].idxmax()]

        min_point = \
            df.loc[(df['Standardkraftsensor'] < 1) & (df['Prüfzeit'] > max_point['Prüfzeit'])].iloc[0]

        max_distance = df.loc[df['Standardweg'].idxmax()]

        springback = max_point['Standardwegsensor'] - min_point['Standardwegsensor']
        distance = max_distance['Standardweg']

        y_values.append(springback)
        x_values.append(distance)

    plt.plot(x_values, y_values, marker='o', linestyle='None', color='red', label='Single bend')
    plt.grid(True)
    plt.savefig('single_bend_tests2.png')


def delete_unneeded_lines():
    # Get all file names from directory
    file_names = os.listdir('../data/cycle_vs_single_test/single_measurements/tra')

    for file_name in file_names:
        # Read file and delete the first 163 lines (they are not needed) and save acii encoding western european
        with open(f'data/cycle_vs_single_test/single_measurements/tra/{file_name}', 'r', encoding='cp1252') as f:
            lines = f.readlines()
            lines = lines[163:]
            # Create dataframe from lines split by ';'
            df = pd.DataFrame([line.split(';') for line in lines])
            # Remove second row from dataframe (it contains the units)
            df = df.drop(df.index[1])
            # Replace ',' with '.' in dataframe
            df = df.replace(',', '.', regex=True)
            # Save dataframe to csv file with ';' as delimiter and '.' as decimal without column numbers
            df.to_csv(f'data/cycle_vs_single_test/single_measurements/cleared/{file_name}.csv', sep=';', decimal='.',
                      index=False)

        # Open file and delete the first row (it contains the units) save it again
        # Workaround because pandas add a row ....
        with open(f'data/cycle_vs_single_test/single_measurements/cleared/{file_name}.csv', 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            # Remove '"' from lines
            lines = [line.replace('"', '') for line in lines]
            # Delete empty lines
            lines = [line for line in lines if line != '\n']
            with open(f'data/cycle_vs_single_test/single_measurements/cleared/{file_name}.csv', 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    # read_excel()j
    plot_all_springbacks()
    # delete_unneeded_lines()
    # plot_the_single_bend(