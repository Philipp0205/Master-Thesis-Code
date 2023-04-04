import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Schritt drei - Modell trainieren
def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')


# Schritt zwei - Daten vorbereiten und in Trainings- und Testdaten aufteilen
def prepare_data(df):
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # random split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Schritt ein - Daten laden
def load_data(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    df = load_data('../../peter_tutorial/data/data.csv')

    X_train, X_test, y_train, y_test = prepare_data(df)
    train_model(X_train, X_test, y_train, y_test)
