import pandas as pd
import numpy as np
import requests
import pathlib
import pickle
import argparse
from itertools import product
from io import StringIO
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, GroupShuffleSplit,\
    StratifiedGroupKFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
# import missingno as msno
from utils import preprocessing
from utils import GridSearchCV_with_groups
from utils import plot_scatterplots
from utils import plot_confusion_matrix
from utils import get_design_matrix
from utils import get_models

parser = argparse.ArgumentParser()
parser.add_argument('data_type', help='which samples and features to use',
                    choices=[
                        'majors_only', 'traces_only', 'majors_or_traces',
                        'majors_and_traces_restricted', 'majors_and_traces'])
parser.add_argument('--n_jobs', help='the number of cores to be used',
                    default=1, type=int)
parser.add_argument('--plot',  help='whether to plot the figures',
                    action='store_true')
args = parser.parse_args()
data_type = args.data_type

# Choose the data to be used for learning.
# data_type = 'majors_only'
# data_type = 'traces_only'
# data_type = 'majors_or_traces'
# data_type = 'majors_and_traces_restricted'
# data_type = 'majors_and_traces'

# Define the root of the output result directory
result_dir = "../results/"
figure_dir = "../figures/"

# Create the directory tree
pathlib.Path(f"{result_dir}/{data_type}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{figure_dir}/{data_type}").mkdir(parents=True, exist_ok=True)

# Download the dataset from the ESPRI server as .csv
url = "https://data.ipsl.fr/repository/TephraDatabase/TephraDataBase.csv"
r = requests.get(url)
r.encoding = "utf-8"
s = r.text
df = pd.read_csv(StringIO(s), low_memory=False)

# in case the ESPRI server is down:
# df = pd.read_csv(
# "..assets/Data/BOOMDataset.csv", encoding = 'UTF-8', low_memory =False )

# Data preprocessing. For more info on the preprocessing of data, the reader is
# referred to the BOOMDataset_preprocessing notebook
df = preprocessing(df)

# Filter samples for which volcano ID is unknown.
df_volcanoes = df.loc[df.Volcano != 'Unknown'].copy()

# X_volcanoes only contains the columns related to the desired chemical
# elements
# df_volcanoes and X_volcanoes are restricted to the rows for which the
# desired chemical elements are not all missing values.
df_volcanoes, X_volcanoes = get_design_matrix(
    df_volcanoes, data_type
)

# Define target attributes (volcano) and list of SampleIDs to consider in the
# train/test splits.
df_volcanoes['Volcano'] = df_volcanoes['Volcano'].astype("category")
df_volcanoes['SampleID'] = df_volcanoes['SampleID'].astype("category")
# convert stringgs to interger codes
y = np.array(df_volcanoes['Volcano'].cat.codes, dtype='int8')
SampleID = np.array(df_volcanoes['SampleID'].cat.codes)
# Keep track of code-strings correspondance
volcano_list = np.array(df_volcanoes['Volcano'].cat.categories)
SampleID_list = np.array(df_volcanoes['SampleID'].cat.categories)

# A few sanity checks
# -------------------
unique, counts = np.unique(SampleID, return_counts=True)
n_sampleID = len(counts)
print(f'There are {n_sampleID} unique sampleIDs:')

# Print number of sample observations per class
unique, counts = np.unique(y, return_counts=True)
n_classes = len(counts)
print(f'There are {n_classes} volcans distributed as follows:')
for u, c in zip(unique, counts):
    print(f'id: {u}, volcán: \
          \033[1m{volcano_list[u]}\033[0m, sample observations: {c}'
          )

# Count nb of samples per volcan.
ct = pd.crosstab(SampleID, y)
ct.sum(axis=0)

# Visualize the missing data
# msno.matrix(X_volcanoes)
# plt.show()

print('Percentage of missing values for each elements')
X_volcanoes.isna().mean().round(4) * 100

# Define the models to be trained
# -------------------------------
imputations, models, grids = get_models()

imputation_names = list(imputations.keys())
model_names = list(models.keys())

if data_type == 'majors_only' or data_type == 'majors_and_traces_restricted':
    # If only majors are used, there are very few missing values so a simple
    # imputation will be used.
    model_names.remove('gb')
    model_components = [('mean_imp', model_name) for model_name in model_names]
    model_components.append((None, 'gb'))  # gb does not require imputation
else:
    model_names.remove('gb')
    model_components = list(product(imputation_names, model_names))
    model_components.append((None, 'gb'))  # gb does not require imputation


def get_clf_grid(comps):
    imp, mod = comps
    if imp is not None:
        clf = make_pipeline(
            clone(imputations[imp]), StandardScaler(), clone(models[mod])
        )
        clf_name = '_'.join(comps)
    else:
        clf = make_pipeline(StandardScaler(), clone(models[mod]))
        clf_name = mod
    grid = grids[mod]

    return clf, grid, clf_name


# Run cross-validation with grid search for all models.
results = {}
for comps in model_components:
    print(comps)

    clf, grid, clf_name = get_clf_grid(comps)

    est = GridSearchCV_with_groups(
        clf, grid, cv_test_size=0.2, cv_n_splits=5, n_jobs=args.n_jobs)

    # gss = GroupShuffleSplit(test_size=0.2, n_splits=5, random_state=0)
    gss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)

    cv = cross_validate(est,
                        X_volcanoes,
                        y,
                        groups=SampleID,
                        scoring=['accuracy', 'balanced_accuracy'],
                        fit_params={'groups': SampleID},
                        cv=gss,
                        return_estimator=True,
                        n_jobs=1)

    results[clf_name] = cv

# Save results
model_names = list(results.keys())
measures = ['fit_time', 'score_time', 'test_accuracy',
            'test_balanced_accuracy']
for measure in measures:
    res = pd.DataFrame([results[c][measure] for c in model_names])
    res = res.transpose()
    res.columns = model_names
    res.to_csv(f'{result_dir}/{data_type}/{measure}.csv')

best_params = {}
for c in model_names:
    param_names = list(results[c]['estimator'][0].best_params_.keys())
    best_params[c] = {k: [] for k in param_names}
    for est in results[c]['estimator']:
        for k in param_names:
            best_params[c][k].append(est.best_params_[k])
file = open(f'{result_dir}/{data_type}/best_params.pkl', 'wb')
pickle.dump(best_params, file)
file.close()

if args.plot:
    # Visualize results for the first fold
    for comps in model_components:
        clf, grid, clf_name = get_clf_grid(comps)

        # recover the first outer cross-validation split
        gss = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
        train_out, test_out = next(
            gss.split(X_volcanoes, y=y, groups=SampleID)
        )
        # recover the fitted predictor for this split
        est = results[clf_name]['estimator'][0].best_estimator_

        X_test_out = X_volcanoes.iloc[test_out]
        y_test_out = y[test_out]
        pred = est.predict(X_test_out)
        volcano_list = df_volcanoes.Volcano.cat.categories

        # plot imputation and classification in different scatter plots in
        # order to have an idea of which samples are badly classified
        plot_scatterplots(X_test_out, y_test_out,
                        'SiO2_normalized', 'K2O_normalized',
                        est, pred, volcano_list, dir=figure_dir,
                        name=f'{data_type}/{clf_name}', save=True)
        if 'traces' in data_type:
            plot_scatterplots(X_test_out, y_test_out, 'SiO2_normalized', 'La',
                            est, pred, volcano_list, dir=figure_dir,
                            name=f'{data_type}/{clf_name}', save=True)
            plot_scatterplots(X_test_out, y_test_out, 'Rb', 'Sr',
                            est, pred, volcano_list, dir=figure_dir,
                            name=f'{data_type}/{clf_name}', save=True)
            plot_scatterplots(X_test_out, y_test_out, 'Cs', 'La',
                            est, pred, volcano_list, dir=figure_dir,
                            name=f'{data_type}/{clf_name}', save=True)

        # plot the confusion matrix
        volcanoes_by_latitude = np.asarray(
            ['Llaima', 'Sollipulli', 'Caburga-Huelemolle', 'Villarrica',
            'Quetrupillán', 'Lanín', 'Huanquihue Group', 'Mocho-Choshuenco',
            'Carrán-Los Venados', 'Puyehue-Cordón Caulle',
            'Antillanca-Casablanca', 'Osorno', 'Calbuco', 'Yate', 'Apagado',
            'Hornopirén', 'Huequi', 'Michinmahuida', 'Chaitén', 'Corcovado',
            'Yanteles', 'Melimoyu', 'Mentolat', 'Cay'
            'Macá', 'Hudson', 'Lautaro', 'Viedma', 'Aguilera', 'Reclus',
            'Monte Burney'])

        y_test_names = volcano_list[y_test_out]
        y_pred_names = volcano_list[pred]
        plot_confusion_matrix(
            y_test_names, y_pred_names, labels=volcanoes_by_latitude,
            name=f'{data_type}/ConfusionMatrix_{clf_name}', dir=figure_dir,
            save=True
        )

        # plot permutation importance
        result = permutation_importance(
            est, X_test_out, y_test_out, n_repeats=10, random_state=42,
            n_jobs=3
        )
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.boxplot(result.importances[sorted_idx].T, vert=False,
                labels=X_test_out.columns[sorted_idx]
                )
        ax.set_title("Permutation Importances (test set)")
        fig.tight_layout()
        plt.savefig(
            f'{figure_dir}/{data_type}/permutation_importance_{clf_name}.png',
            dpi=300, bbox_inches='tight')
        plt.close()
