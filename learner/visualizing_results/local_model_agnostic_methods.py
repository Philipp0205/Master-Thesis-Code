import numpy as np
from matplotlib import pyplot as plt
import learner.data_preprocessing as preprocessing

from lime import lime_tabular


def create_lime_explanation(name, model, df, md):
    feature_names = df.columns
    feature_names = np.array(feature_names)

    predict_fn_rf = lambda x: model.predict(x).astype(float)
    X = md.X_train.values

    explainer = lime_tabular.LimeTabularExplainer(X, feature_names=md.X_train.columns,
                                                  class_names=['springback'],
                                                  kernel_width=5,
                                                  mode='regression',
                                                  )

    # Choose instance to explain
    instance = md.X_test.iloc[1, :]

    exp = explainer.explain_instance(instance, predict_fn_rf, num_features=10)
    plt.style.use(['science', 'ieee'])

    root = preprocessing.get_root_directory()
    fig = exp.as_pyplot_figure()
    fig.savefig(f'{root}/learner/visualizing_results/results/lime/lime_{name}_pyplot.png')
    exp.save_to_file(f'{root}/learner/visualizing_results/results/lime/lime_{name}.html')
