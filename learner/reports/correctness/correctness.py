import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import learner.data_preprocessing as pre
import scienceplots


def calculate_correctness(model, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def performance_test_train_val(model, model_data):
    y_pred_train = model.predict(model_data.X_train)
    y_pred_test = model.predict(model_data.X_test)
    y_pred_val = model.predict(model_data.X_val)

    mae_train, mse_train, rmse_train = calculate_correctness(
        model,
        model_data.y_train,
        y_pred_train
    )

    mae_test, mse_test, rmse_test = calculate_correctness(
        model,
        model_data.y_test,
        y_pred_test
    )

    mae_val, mse_val, rmse_val = calculate_correctness(
        model,
        model_data.y_val,
        y_pred_val
    )

    return rmse_train, rmse_test, rmse_val


def performance_plot(df):
    root = pre.root_directory()

    # Create plot for each row in df with the columns 'name' and 'train', 'test', 'val'
    # and the rows 'rmse', 'mae', 'mse', 'r2'.
    for index, row in df.iterrows():
        plt.style.use(['science', 'ieee'])

        y = [row['train'], row['val'], row['test']]
        x = ['train', 'validation', 'test']

        # Annotate the test rmse value
        plt.annotate(f'{row["test"]:.2f}', xy=(2, row['test']),
                     xytext=(2.12, row['test'] - 0.005))

        # give the annotation the same color as the line
        plt.annotate('', xy=(2, row['test']), xytext=(2.1, row['test']),
                     arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

        # latex y label
        plt.ylabel(r'$\delta\alpha_{SB}$')
        plt.xlabel('')

        plt.plot(x, y, label=row['name'], marker='o', markersize=2)

    plt.legend()
    plt.savefig(f'{root}/learner/reports/correctness/performance_plot.png', dpi=600,
                transparent=True)

def correctness_report(name, model_data, model, y_pred):
    print('------- CORRECTNESS REPORT --------')
    # print(f'Model: {model.steps}\n')

    # mae, mse, rmse = calculate_correctness(
    #     model,
    #     model_data.y_test,
    #     y_pred
    # )

    rmse_train, rmse_test, rmse_val = performance_test_train_val(model, model_data)

    # Create dataframe with the results with the columns 'name' and 'train', 'test', 'val'
    # and the rows 'rmse', 'mae', 'mse', 'r2'.
    df = pd.DataFrame(columns=['name', 'train', 'test', 'val'])

    # Concat the name and the results to the dataframe
    # df = pd.concat([df, pd.DataFrame(
    #     {'name': name, 'train': rmse_train, 'test': rmse_test, 'val': rmse_val},
    #     index=[0])])

    root = pre.root_directory()

    # Get csv as dataframe
    df = pd.read_csv(f'{root}/learner/reports/correctness/correctness_report.csv')

    # Concat the name and the results to the dataframe
    # When the dataframe does not contain the name of the model, add it to the dataframe
    if name not in df['name'].values:
        df = pd.concat([df, pd.DataFrame(
            {'name': name, 'train': rmse_train, 'test': rmse_test, 'val': rmse_val},
            index=[0])])

    # Save the dataframe to a csv file
    df.to_csv(f'{root}/learner/reports/correctness/correctness_report.csv', index=False)
    performance_plot(df)

    print('\n------- END CORRECTNESS REPORT --------\n')
