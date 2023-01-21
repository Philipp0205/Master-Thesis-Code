import RegscorePy as rs


def get_number_of_parameters(model):
    # Get the number of parameters of the pipeline
    params = model.steps[1][1].get_params(deep=True)

    return len(params)


def interpretability_report(model_data, model, y_pred):
    print('------- INTERPRETABILITY REPORT --------')
    # print(f'Model: {model.steps}\n')
    params = get_number_of_parameters(model)

    print(f'Number of parameters: {params}')

    print('\n------- END INTERPRETABILITY REPORT --------\n')
