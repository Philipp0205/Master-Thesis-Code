\usepackage{ulem}import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib
import scienceplots
import matplotlib.patches as mpatch
import numpy as np
import learner.data_preprocessing as preprocessing


def visualize_dataset():
    fig = plt.figure()
    x = np.arange(10)

    # Load conolidated dataset
    # dataset = pd.read_csv('data/dataset/data.csv', delimiter=',')

    dataset = preprocessing.get_data()
    print(dataset.head())

    # Get thickness and die_opening for each row
    thicknesses_and_die_openings = dataset[['thickness', 'die_opening']]

    # Remove duplicate rows from thicknesses_and_die_openings
    thicknesses_and_die_openings = thicknesses_and_die_openings.drop_duplicates()

    # Select samples where die_opening is 30
    test_data = thicknesses_and_die_openings[thicknesses_and_die_openings['die_opening'] == 30]
    train_data = thicknesses_and_die_openings[thicknesses_and_die_openings['die_opening'] != 30]

    x_test = test_data['thickness']
    y_test = test_data['die_opening']

    x_train = train_data['thickness']
    y_train = train_data['die_opening']

    # x = thicknesses_and_die_openings['thickness']
    # y = thicknesses_and_die_openings['die_opening']

    # Plot the dataset
    plt.style.use(['science', 'scatter', 'grid', 'ieee'])

    # Plot the dataset with seaborn with diamond markers
    plt.figure(dpi=1000)

    # seaborn.scatterplot(x=x_test, y=y_test, marker='D', s=40, color='blue', label='Test')
    # seaborn.scatterplot(x=x_train, y=y_train, marker='D', s=40, color='lightcoral', label='Train')

    sn.scatterplot(x=x_train, y=y_train, marker='D', s=40, label='Train')
    sn.scatterplot(x=x_test, y=y_test, marker='D', color='blue', s=40, label='Test')

    # rectangle_test = plt.Rectangle((0.4, 26), 2.7, 8, linestyle='dashed', fc='none', ec="red",
    #                                label='Test',
    #                                linewidth=0.8)
    # rectangle_train = plt.Rectangle((0.4, 8.5), 2.7, 17, linestyle='dashed', fc='none', ec="blue",
    #                                 label='Train',
    #                                 linewidth=0.8)
    #
    # rectangle_train2 = plt.Rectangle((0.4, 34.6), 2.7, 16.9, linestyle='dashed', fc='none',
    #                                  ec="blue",
    #                                  linewidth=0.8)

    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

    # plt.gca().add_patch(rectangle_test)
    # plt.gca().add_patch(rectangle_train)
    # plt.gca().add_patch(rectangle_train2)

    # q: How to make the legend transparent?
    # a: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                        fancybox=True, ncol=5)

    plt.legend(frameon=False)

    legend = plt.legend()
    legend.get_frame().set_edgecolor('b')
    legend.get_frame().set_linewidth(0.0)

    # Place legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.xlabel('Thickness $t$ [mm]')
    plt.ylabel('Die opening $V$ [mm]')

    root = preprocessing.get_root_directory()

    plt.savefig('results/test_train_split.png', transparent=True)

    plt.clf()


def visualize_mean_springback(df):
    # Plot a heatmap with seaborn with the mean springback for each thickness/die_opening combination
    plt.figure(dpi=1000)
    plt.style.use(['science', 'grid'])

    # Get thickness and die_opening for each row
    thicknesses_and_die_openings = df[['thickness', 'die_opening']]

    # Remove duplicate rows from thicknesses_and_die_openings
    thicknesses_and_die_openings = thicknesses_and_die_openings.drop_duplicates()

    # Get the mean springback for each thickness/die_opening combination
    mean_springback = df.groupby(['die_opening', 'thickness']).mean()['springback']

    # Reshape the mean_springback series into a dataframe
    mean_springback = mean_springback.unstack()

    # Plot the mean_springback dataframe with seaborn
    sn.heatmap(mean_springback, annot=True, fmt='.2f', cmap='viridis')

    # Write the V/t ratio below the mean spring back
    for i in range(0, len(mean_springback.index)):
        for j in range(0, len(mean_springback.columns)):
            # Get the die_opening and thickness for the current cell
            die_opening = mean_springback.index[i]
            thickness = mean_springback.columns[j]
            # Get the mean springback for the current cell
            springback = mean_springback.iloc[i, j]
            # Get the V/t ratio for the current cell
            ratio = die_opening / thickness
            # Write the V/t ratio below the mean springback
            plt.text(j+0.5, i+0.22, f'({round(ratio, 2)})', ha="center", va="bottom", color="w")

    # Change x and y of heat map to be the thickness and die_opening
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    # Sort y axsis from low to high
    plt.gca().invert_yaxis()

    plt.ylabel('Die opening $V$ [mm]')
    plt.xlabel('Thickness $t$ [mm]')

    plt.savefig('results/mean_springback_heatmap.png', transparent=True)


if __name__ == '__main__':
    # visualize_dataset()
    data = preprocessing.get_data()
    visualize_mean_springback(data)
