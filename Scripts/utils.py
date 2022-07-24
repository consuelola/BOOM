import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
import seaborn as sns
from functions import colores
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class GridSearchCV_with_groups():
    def __init__(self, estimator, param_grid, cv_test_size, cv_n_splits,
                 n_jobs=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_test_size = cv_test_size
        self.cv_n_splits = cv_n_splits
        self.n_jobs = n_jobs

    def fit(self, X, y, groups):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.core.series.Series):
            y = y.to_numpy()

        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [
            dict(zip(keys, combination)) for combination in product(*values)]
        
        print(combinations)

        gss = GroupShuffleSplit(
            test_size=self.cv_test_size, n_splits=self.cv_n_splits)

        scores = np.empty((self.cv_n_splits, len(combinations)))

        for i, (train, test) in enumerate(gss.split(X, groups=groups)):
            X_train_in = X[train]
            X_test_in = X[test]
            y_train_in = y[train]
            y_test_in = y[test]

            for j, comb in enumerate(combinations):
                self.estimator.set_params(**comb)
                self.estimator.fit(X_train_in, y_train_in)
                scores[i, j] = self.estimator.score(X_test_in, y_test_in)

        median_score = np.median(scores, axis=0)
        best_comb = np.argmax(median_score)

        self.scores_ = scores
        self.params_ = combinations
        self.best_params_ = combinations[best_comb]
        self.best_score_ = median_score[best_comb]

        # Refit on the whole dataset with the best paraps
        self.best_estimator_ = clone(
            self.estimator.set_params(**self.best_params_))
        self.best_estimator_.fit(X, y)
    
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


def plot_scatterplots(X_volcanoes, yv, X_test_out, yv_test_out, best_est, pred,
                      volcano_list, target_type='volcano'):

    ind_wrong = pred != yv_test_out

    X_test_imp = Pipeline(best_est.steps[:-1]).transform(X_test_out)
    X_test_imp = pd.DataFrame(X_test_imp, columns=X_test_out.columns)

    # Plot Original data (how are missing values treated?)
    # Only fully observed points plotted?
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    A = 'SiO2_normalized'
    B = 'K2O_normalized'
    yv_names = volcano_list[yv]
    sns.scatterplot(
        x=X_volcanoes.loc[:, A], y=X_volcanoes.loc[:, B],
        hue=yv_names, alpha=0.7,
        palette=colores(yv_names, target_type), ax=axes[0]
    )
    axes[0].set_title("Original data")
    axes[0].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)

    # Plot Imputed data with ground truth labels
    yv_test_names = volcano_list[yv_test_out]
    sns.scatterplot(
        x=X_test_imp.loc[:, A], y=X_test_imp.loc[:, B],
        hue=yv_test_names, alpha=0.7,
        palette=colores(yv_test_names, target_type), ax=axes[1])
    sns.scatterplot(
        x=X_test_imp.loc[ind_wrong, A],
        y=X_test_imp.loc[ind_wrong, B],
        ax=axes[1], marker='x', color='k', s=30
    )
    axes[1].set_title(
        "Imputed and normalized test data \n with ground truth labels")
    axes[1].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)

    # Plot Imputed data with predicted labels
    yv_pred_names = volcano_list[pred]
    sns.scatterplot(
        x=X_test_imp.loc[:, A],  y=X_test_imp.loc[:, B],
        hue=yv_pred_names, alpha=0.7,
        palette=colores(yv_pred_names, target_type), ax=axes[2]
    )
    sns.scatterplot(
        x=X_test_imp.loc[ind_wrong, A],
        y=X_test_imp.loc[ind_wrong, B],
        ax=axes[2], marker='x', color='k', s=30
    )
    axes[2].set_title(
        "Imputed and normalized test data \n with predicted labels")
    axes[2].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)


def plot_confusion_matrix(yv_test_names, yv_pred_names, labels):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(
        yv_test_names, yv_pred_names, labels=labels, normalize='true'
    )
    cm = (cm.T/cm.sum(axis=1)).T
    plt.imshow(cm, cmap='viridis')
    plt.colorbar()
    n_volcanoes = len(labels)
    ax.set_xticks(np.arange(n_volcanoes))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(n_volcanoes))
    ax.set_yticklabels(labels)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(rotation=90)
    fig.show()


# # Cross-validation without grid search
# # (i.e we take the default hyperparameters of the models)
# # ------------------------------------------------------
# def cross_val_without_gridsearch(est, imp):
#     start = time.time()
#     balanced_scores = []
#     scores = []

#     # Split test and train sets making sure that Sample Obersvations with the same SampleID 
#     # are not separated
#     gss = GroupShuffleSplit(test_size=.30, n_splits=5, random_state = 0)
#     for train_index, test_index in gss.split(X_volcanoes, groups=SampleID_volcanoes):
#         X_train_volcanoes, X_test_volcanoes = X_volcanoes.iloc[train_index], X_volcanoes.iloc[test_index]
#         yv_train, yv_test = yv[train_index], yv[test_index]

#         #imp = KNNImputer(n_neighbors=2)
#         T_train_imp_volcanoes = imp.fit_transform(X_train_volcanoes)
#         T_test_imp_volcanoes = imp.transform(X_test_volcanoes)

#         sc = StandardScaler()
#         T_train_esc_volcanoes = sc.fit_transform(T_train_imp_volcanoes)
#         T_test_esc_volcanoes = sc.transform(T_test_imp_volcanoes)

#         est.fit(T_train_esc_volcanoes, yv_train)
#         ac = est.score(T_test_esc_volcanoes, yv_test)
#         scores.append(ac)
#         bc = balanced_accuracy_score(yv_test, est.predict(T_test_esc_volcanoes))
#         balanced_scores.append(bc)
#     end = time.time()
#     runtime = end - start
    
#     return scores, balanced_scores, runtime