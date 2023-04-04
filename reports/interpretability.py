import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import scienceplots


def partial_dependence_plot(model, X, features):
    # Create PD plot for each feature and create a subplot for each
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Decision Tree")
    disp = PartialDependenceDisplay.from_estimator(model, X,
                                                   ["die_opening", "thickness", "distance"],
                                                   ax=ax)

    plt.savefig('partial_dependence_plot.png', transparent=True)


def get_number_of_parameters(model):
    # Get the number of parameters of the pipeline
    params = model.steps[1][1].get_params(deep=True)

    return len(params)


def interpretability_report(model_data, model, y_pred):
    print('------- INTERPRETABILITY REPORT --------')
    # print(f'Model: {model.steps}\n')
    params = get_number_of_parameters(model)

    # partial_dependence_plot(model, model_data.X, [])

    print(f'Number of parameters: {params}')

    print('\n------- END INTERPRETABILITY REPORT --------\n')
