import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


# 1. Datensatz laden
def load_data(path):
    return pd.read_csv(path)


# 2. Daten vorbereiten: In Test- trainingsdaten aufteilen
def prepare_data(df):
    X = df[['distance', 'thickness', 'die_opening']]
    y = df['springback']

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# 3. Modell trainieren und testen
def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42, n_estimators=300)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'RMSE {rmse:.2f}')


def grid_search(X_train, y_train):
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        # 'max_depth': [5, 10, 15],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 4]
    }

    gs = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print(f'Best parameters: {gs.best_params_}')







if __name__ == '__main__':
    df = load_data('../../peter_tutorial/data/data.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    # train_model(X_train, X_test, y_train, y_test)
    grid_search(X_train, y_train)