import time

import profiler as profiler
# import memeory usage from scikit learn
from memory_profiler import memory_usage


def calculate_trainig_time(model_data, model, y_pred):
    X = model_data.X
    y = model_data.y

    start = time.time()
    model.fit(X, y)
    elapsed_time = time.time() - start

    return elapsed_time * 1000


def calculate_inference_time(model_data, model, y_pred):
    X_test = model_data.X_test

    sample = X_test.sample(n=100)

    start = time.time()
    model.predict(sample)
    end = time.time()

    return (end - start) * 1000


def calculate_memory_usage(model_data, model, y_pred):
    X_train = model_data.X_train
    y_train = model_data.y_train

    mem = memory_usage((model.fit, (X_train, y_train)), max_usage=True)
    return mem


def resource_report(model_data, model, y_pred):
    print('------- RESOURCE REPORT --------')

    training_time = calculate_trainig_time(model_data, model, y_pred)
    inference_time = calculate_inference_time(model_data, model, y_pred)
    memory_usage = calculate_memory_usage(model_data, model, y_pred)

    print(f'Training time: {round(training_time, 3)} s')
    print(f'Inference time: {round(inference_time, 3)} s')
    print(f'Memory usage: {round(memory_usage, 3)} MB')

    print('\n------- END RESOURCE REPORT --------\n')
