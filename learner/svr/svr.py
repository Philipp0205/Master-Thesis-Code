import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr():
    # get the dataset
    dataset = pd.read_csv('data/data.csv')
    # our dataset in this implementation is small, and thus we can print it all instead of viewing only the end
    # print(dataset)

    # split the data into features and target variable separatly
    X_l = dataset.iloc[:, 1:-1].values  # features set
    y_p = dataset.iloc[:, -1].values  # set of study variable

    # Reshape the salary from 1D to 2D model
    y_p = y_p.reshape(-1, 1)

    # Scale up the X_l and y_p variables separately as shown:
    # As we can see from the obtained output, both variables were scaled within the range -3 and +3.
    StdS_X = StandardScaler()
    StdS_y = StandardScaler()
    X_l = StdS_X.fit_transform(X_l)
    y_p = StdS_y.fit_transform(y_p)

    # print("Scaled X_l:")
    # print(X_l)
    # print("Scaled y_p:")
    # print(y_p)
    #
    # plt.scatter(X_l, y_p, color='red')  # plotting the training set
    # plt.title('Scatter Plot')  # adding a tittle to our plot
    # plt.xlabel('Levels')  # adds a label to the x-axis
    # plt.ylabel('Salary')  # adds a label to the y-axis
    # plt.savefig('results/levels_salary_plot.png')  # saves the plot as a png file

    # The plot shows a non-linear relationship between the Levels and Salary.
    # create the model object
    regressor = SVR(kernel='rbf')
    # fit the model on the data
    regressor.fit(X_l, y_p.ravel())

    # Since the model is now ready, we can use it and make predictions as shown:
    A = regressor.predict(StdS_X.transform([[6.5]]))
    print(A)

    # Convert A to 2D
    A = A.reshape(-1, 1)
    print(A)

    # Taking the inverse of the scaled value
    A_pred = StdS_y.inverse_transform(A)
    print(A_pred)

    B_pred = StdS_y.inverse_transform(regressor.predict(StdS_X.transform([[6.5]])).reshape(-1, 1))
    print(B_pred)

    # inverse the transformation to go back to the initial scale
    plt.scatter(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(y_p), color='red')
    plt.plot(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(regressor.predict(X_l).reshape(-1, 1)),
             color='blue')
    # add the title to the plot
    plt.title('Support Vector Regression Model')
    # label x axis
    plt.xlabel('Position')
    # label y axis
    plt.ylabel('Salary Level')
    # print the plot
    plt.savefig('results/svr_plot.png')


def svr_with_one_test_row():
    project_root = get_project_root()

    input_directory = project_root / 'data' / 'dataset'
    output_directory = project_root / 'learner' / 'svr' / 'results'
    dirs = [x[0] for x in os.walk(input_directory)]

    for d in dirs:
        if 'csv' in d:
            parent_name = Path(d).parent.name
            dataset = pd.read_csv(f'{d}/springbacks.csv')

            x = dataset['distance']
            y = dataset['springback']

            # plt.scatter(x, y, color='red')
            # plt.savefig(f'{output_directory}/{parent_name}_springbacks_plot.png')
            # plt.clf()

            StdS_X = StandardScaler()
            StdS_y = StandardScaler()
            X_l = StdS_X.fit_transform(x.values.reshape(-1, 1))
            y_p = StdS_y.fit_transform(y.values.reshape(-1, 1))

            plt.scatter(X_l, y_p, color='red')  # plotting the training set
            plt.title('Scatter Plot')  # adding a tittle to our plot
            plt.xlabel('Levels')  # adds a label to the x-axis
            plt.ylabel('Salary')  # adds a label to the y-axis
            plt.savefig(f'{output_directory}/{parent_name}_springbacks_plot_scaled.png')
            # clear
            plt.clf()

            # create the model object
            regressor = SVR(kernel='rbf')
            # fit the model on the data
            regressor.fit(X_l, y_p)

            A = regressor.predict(StdS_X.transform([[6.5]]))
            A = A.reshape(-1, 1)
            A_pred = StdS_y.inverse_transform(A)

            B_pred = StdS_y.inverse_transform(regressor.predict(StdS_X.transform([[6.5]])).reshape(-1, 1))

            # inverse the transformation to go back to the initial scale
            plt.scatter(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(y_p), color='red')
            plt.plot(StdS_X.inverse_transform(X_l), StdS_y.inverse_transform(regressor.predict(X_l).reshape(-1, 1)),
                     color='blue')
            # add the title to the plot
            plt.title('Support Vector Regression Model')
            # label x axis
            plt.xlabel('Position')
            # label y axis
            plt.ylabel('Salary Level')
            # print the plot
            plt.savefig(f'{output_directory}/{parent_name}_svr_plot.png')
            plt.clf()


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == '__main__':
    # svr()
    svr_with_one_test_row()
