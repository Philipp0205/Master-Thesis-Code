import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_correctness(model, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def correctness_report(model_data, model, y_pred):
    print('------- CORRECTNESS REPORT --------')
    # print(f'Model: {model.steps}\n')

    mae, mse, rmse = calculate_correctness(
        model,
        model_data.y_test,
        y_pred
    )

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

    print('\n------- END CORRECTNESS REPORT --------\n')
