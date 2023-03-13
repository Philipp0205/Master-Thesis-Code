import matplotlib.pyplot as plt
import numpy as np
from lime import lime_tabular
import learner.data_preprocessing as preprocessing
import learner.mlp.mlp as mlp
import learner.random_forest.random_forest as rf


def create_lime_subplot(md):
    models = [
        ('rf', rf.random_forest(md)),
        ('mlp', mlp.mlp(md.X_train, md.y_train, md.X_test, md.y_test))
    ]

    plt.style.use(['science', 'ieee'])
    fig, axs = plt.subplots()

    # Create a subplot for each model with a lime explanation
    for i in range(len(models)):
        name, model = models[i]
        model = model[0]
        feature_names = md.X_train.columns
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
        axs[i].imshow(exp.as_pyplot_figure())
        axs[i].set_title(name)

    plt.savefig('lime_subplots.png', dpi=600)


