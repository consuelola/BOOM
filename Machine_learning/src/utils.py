import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import GroupShuffleSplit,  StratifiedGroupKFold
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
     HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def preprocessing(df):

    # 1. First of all, we drop rows corresponding to samples not analyzed for
    # geochemistry, as well as outliers, samples for which the volcanic source
    # is uncertain, and samples with Analytical Totals lower than 94 wt.%, 
    # as they might correspond to altered samples.

    is_register = df.TypeOfRegister.isin(
        ['Pyroclastic material', 'Effusive material'])
    isnot_outlier = df.Flag.str.contains(
        'Outlier', na=False, case=False) == False
    isnot_VolcanicSourceIssue = df.Flag.str.contains(
        'VolcanicSource_Issue', na=False, case=False) == False
    df.SiO2 = df.SiO2.replace(np.nan, -1)
    isnot_altered = ((df.Total > 95) & (df.SiO2 != -1)) | (df.SiO2 == -1)
    df.SiO2 = df.SiO2.replace(-1, np.nan)
    df = df.loc[
        is_register & isnot_outlier & isnot_VolcanicSourceIssue & isnot_altered
        ]
    n, _ = df.shape
    # print(f'There are {n} rows left.')

    # 2. In second place, we will replace some of the values in the Dataset.
    # 2.1 Replace element concentrations registered as "0" with
    # "below detection limit" (bdl).
    # Because a value equal to zero is not possible to determine with the
    # current analytical techniques, thus bdl is more accurate.

    for elemento in ["SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3",
                     "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5",
                     "Cl", 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                     'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                     'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                     'Pb', 'Th', 'U']:
        df[elemento] = df[elemento].replace(to_replace=0, value='bdl')

    # 2.2 Replace the various missing values placeholders by np.nan
    df.replace(to_replace='n.a.', value=np.nan, inplace=True)
    df.replace(to_replace='Not analyzed', value=np.nan, inplace=True)
    df.replace(to_replace='-', value=np.nan, inplace=True)
    df.replace(to_replace='Not determined', value=np.nan, inplace=True)
    df.replace(to_replace='n.d', value=np.nan, inplace=True)
    df.replace(to_replace='n.d.', value=np.nan, inplace=True)
    df.replace(to_replace='<0.01', value=np.nan, inplace=True)
    df.replace(to_replace='<0.1', value=np.nan, inplace=True)
    df.replace(to_replace='<1', value=np.nan, inplace=True)
    df.replace(to_replace='<5', value=np.nan, inplace=True)
    df.replace(to_replace='<6', value=np.nan, inplace=True)
    df.replace(to_replace='<10', value=np.nan, inplace=True)
    df.replace(to_replace='Over range', value=np.nan, inplace=True)
    df.replace(to_replace='bdl', value=np.nan, inplace=True)

    # 2.3 Make sure major and trace elements correspond to numbers and not
    # strings.
    df.loc[:, 'Rb':'U'] = df.loc[:, 'Rb':'U'].astype('float')
    df.loc[:, 'SiO2_normalized':'K2O_normalized'] = df.loc[
        :, 'SiO2_normalized':'K2O_normalized'].astype('float')

    # 3. Because Fe can be analyzed in different states
    # (FeO, Fe2O3, FeOT, Fe2O3T), the columns describing Fe have many missing
    # values
    # but which can be filled by transforming one form of Fe into another.
    # Because most of the samples in the BOOM dataset have been
    # analyzed by Electron Microscopy which analyzes Fe as FeOT,
    # we calculate FeOT for all the samples and drop the other rows
    # (Fe2O3, Fe2O3T, FeO) as they are redundant.

    # case 1: Fe is presented as Fe2O3 and FeO in the original publication
    ind = (~df.SiO2_normalized.isna() &
           df.FeOT_normalized.isna() &
           ~df.FeO_normalized.isna() &
           ~df.Fe2O3_normalized.isna() &
           df.Fe2O3T_normalized.isna()
           )
    df.loc[ind, 'FeOT_normalized'] = (
        df.FeO_normalized.loc[ind]+df.Fe2O3_normalized.loc[ind]*0.899)

    # case 2: Fe is presented as Fe2O3T in the original publication
    ind = (~df.SiO2_normalized.isna() &
           df.FeOT_normalized.isna() &
           df.FeO_normalized.isna() &
           ~df.Fe2O3T_normalized.isna() &
           df.Fe2O3_normalized.isna()
           )

    df.loc[ind, 'FeOT_normalized'] = df.Fe2O3T_normalized.loc[ind]*0.899

    df.drop(['FeO_normalized', 'Fe2O3_normalized', 'Fe2O3T_normalized'],
            axis=1, inplace=True)

    # 4. When training the models, all sample observations corresponding to the
    # same sample should either be in the train or test sets.
    # Thus, we will check if there is any volcanic center with information from
    # only one sample ID.
    co = pd.crosstab(df.Volcano, df.SampleID)
    _, n_sampleID = co.shape
    # print(f'There are {n_sampleID} unique samples IDs')
    is_nonzero = co > 0
    n_volcan_per_sampleID = is_nonzero.sum(axis=0)
    unique, counts = np.unique(n_volcan_per_sampleID, return_counts=True)
    ind_ids = np.where(is_nonzero.sum(axis=0) == 2)[0]
    # print(f'There are {len(ind_ids)} sampleIDs which contain several
    # observations from several volcanoes:')
    # print([co.columns[ind_ids].values[i] for i in range(len(ind_ids))])

    n_sampleID_per_volcan = is_nonzero.sum(axis=1)
    ind_ids = np.where(n_sampleID_per_volcan == 1)[0]
    # print(f'There is {len(ind_ids)} volcanic center whose observations all
    # come from the same sample IDs:')
    # print([co.index[i] for i in ind_ids])

    df = df[df.Volcano != co.index[ind_ids[0]]]

    # print(f'There are {len(df)} observations left.')

    # 5. We will drop the volcanoes with less than 10 observations.
    Cay = df.Volcano == 'Cay'
    CordonC = df.Volcano == 'Cordón Cabrera'
    Corcovado = df.Volcano == 'Corcovado'
    Yanteles = df.Volcano == 'Yanteles'

    df = df.loc[~Cay & ~CordonC & ~Corcovado & ~Yanteles]
    n, p = df.shape
    # print(f'The dataset now has {n} samples.')
    return df


class GridSearchCV_with_groups(BaseEstimator, ClassifierMixin):

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
        self.params_ = [
            dict(zip(keys, combination)) for combination in product(*values)]

        # gss = GroupShuffleSplit(
        #     test_size=self.cv_test_size, n_splits=self.cv_n_splits)
        gss = StratifiedGroupKFold(n_splits=self.cv_n_splits, shuffle=True)

        out = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_one)(
                X[train], y[train], X[test], y[test], comb, id_split, id_comb)
            for id_comb, comb in enumerate(self.params_)
            for id_split, (train, test) in enumerate(
                gss.split(X, y=y, groups=groups)
                )
            )

        self.scores_ = np.empty((self.cv_n_splits, len(self.params_)))
        for id_split, id_comb, score in out:
            self.scores_[id_split, id_comb] = score

        median_score = np.median(self.scores_, axis=0)
        best_comb = np.argmax(median_score)

        self.best_params_ = self.params_[best_comb]
        self.best_score_ = median_score[best_comb]

        # Refit on the whole dataset with the best paraps
        self.best_estimator_ = clone(
            self.estimator.set_params(**self.best_params_))

        self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

    def fit_one(self, X_train_in, y_train_in, X_test_in, y_test_in, comb,
                id_split, id_comb):
        est = clone(self.estimator.set_params(**comb))
        est.fit(X_train_in, y_train_in)
        return id_split, id_comb, est.score(X_test_in, y_test_in)


def plot_scatterplots(X_test_out, y_test_out, chem_A, chem_B, est, pred,
                      volcano_list, dir='../figures/', name='default',
                      target_type='volcano', save=False):

    ind_wrong = pred != y_test_out
    if len(est.steps) == 3:
        X_test_imp = Pipeline(est.steps[:-2]).fit_transform(
            X_test_out)
    else:
        # case of Hist Gradient Boosting for which there is no imputation
        # We choose to impute by the mean by default.
        imp = SimpleImputer()
        X_test_imp = imp.fit_transform(X_test_out)
    X_test_imp = pd.DataFrame(X_test_imp, columns=X_test_out.columns)
    y_test_names = volcano_list[y_test_out]

    # Plot Original data (how are missing values treated?)
    # Only fully observed points plotted?
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    sns.scatterplot(
        x=X_test_out.loc[:, chem_A], y=X_test_out.loc[:, chem_B],
        hue=y_test_names, alpha=0.7,
        palette=colores(y_test_names, target_type), ax=axes[0]
    )
    axes[0].set_title("Original data")
    axes[0].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)

    # Plot Imputed data with ground truth labels
    sns.scatterplot(
        x=X_test_imp.loc[:, chem_A], y=X_test_imp.loc[:, chem_B],
        hue=y_test_names, alpha=0.7,
        palette=colores(y_test_names, target_type), ax=axes[1])
    sns.scatterplot(
        x=X_test_imp.loc[ind_wrong, chem_A],
        y=X_test_imp.loc[ind_wrong, chem_B],
        ax=axes[1], marker='x', color='k', s=30
    )
    axes[1].set_title(
        "Imputed and normalized test data \n with ground truth labels")
    axes[1].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)

    # Plot Imputed data with predicted labels
    y_pred_names = volcano_list[pred]
    sns.scatterplot(
        x=X_test_imp.loc[:, chem_A],  y=X_test_imp.loc[:, chem_B],
        hue=y_pred_names, alpha=0.7,
        palette=colores(y_pred_names, target_type), ax=axes[2]
    )
    sns.scatterplot(
        x=X_test_imp.loc[ind_wrong, chem_A],
        y=X_test_imp.loc[ind_wrong, chem_B],
        ax=axes[2], marker='x', color='k', s=30
    )
    axes[2].set_title(
        "Imputed and normalized test data \n with predicted labels")
    axes[2].legend(loc='center left', bbox_to_anchor=(0, -0.65), ncol=2)

    if save:
        plt.savefig(
            dir + name + '_' + chem_A + 'vs' + chem_B + '.png',
            dpi=300, bbox_inches='tight', facecolor='w')
    else:
        plt.show()


def plot_confusion_matrix(y_test_names, y_pred_names, labels, name='default',
                          dir='../figures/', save=False):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(
        y_test_names, y_pred_names, labels=labels, normalize='true'
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
    if save:
        plt.savefig(dir+name+'.png', dpi=300, bbox_inches='tight',
                    facecolor='w')
    else:
        plt.show()


def get_design_matrix(df_volcanoes, data_type):
    # Retrieve the geochemical data. FeO, Fe2O3 and FeO2O3T are dropped because
    # FeOT is a different expression of the same element (Fe).
    # P2O5 and Cl are also dropped because they are sporadically analyzed.
    majors = ['SiO2_normalized', 'TiO2_normalized', 'Al2O3_normalized',
              'FeOT_normalized',
              # 'FeO_normalized', 'Fe2O3_normalized', 'Fe2O3T_normalized',
              'MnO_normalized', 'MgO_normalized', 'CaO_normalized',
              'Na2O_normalized', 'K2O_normalized',
              # 'P2O5_normalized','Cl_normalized'
              ]

    traces = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La',
              'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
              'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb',
              'Th', 'U']

    if data_type == 'majors_and_traces':
        X_major = df_volcanoes.loc[:, majors]
        X_traces = df_volcanoes.loc[:, traces]
        X_volcanoes = pd.concat([X_major, X_traces], axis=1)

    elif data_type == 'majors_only':
        X_major = df_volcanoes.loc[:, majors]
        # Remove rows with only missing values
        mask_major = X_major.isna()
        ind_not_all_na = np.where(
            mask_major.sum(axis=1) != X_major.shape[1])[0]
        X_volcanoes = X_major.iloc[ind_not_all_na, ]
        df_volcanoes = df_volcanoes.iloc[ind_not_all_na, ]

    elif data_type == 'traces_only':
        X_traces = df_volcanoes.loc[:, traces]
        # Remove rows with only missing values
        mask_traces = X_traces.isna()
        ind_not_all_na = np.where(
            mask_traces.sum(axis=1) != X_traces.shape[1])[0]
        X_volcanoes = X_traces.iloc[ind_not_all_na, ]
        df_volcanoes = df_volcanoes.iloc[ind_not_all_na, ]

    else:
        raise ValueError(f'`data_type` is {data_type}. Should be one of\
             majors_and_traces, majors_only or  traces_only')

    return df_volcanoes, X_volcanoes


def get_models():

    # K Nearest Neighbors
    knn = KNeighborsClassifier()
    grid_knn = {'kneighborsclassifier__n_neighbors': [2, 5, 10],
                'kneighborsclassifier__weights': ['uniform', 'distance']}

    # Multinomial logistic regression
    lr = LogisticRegression(multi_class='multinomial', class_weight='balanced')
    grid_lr = {'logisticregression__C': [1e-2, 1e-1, 1, 1e1, 1e2],
               'logisticregression__solver': ['saga']}

    # Random Forest
    # usar bootstrap false o class weight balanced_subsample no tiene gran
    # efecto!
    rf = RandomForestClassifier(class_weight='balanced', bootstrap=False,
                                verbose=0)
    grid_rf = {'randomforestclassifier__n_estimators': [50, 100, 200],
               'randomforestclassifier__min_samples_leaf': [1, 5]}

    # Gradient Boosting
    gb = HistGradientBoostingClassifier()
    grid_gb = {
        'histgradientboostingclassifier__learning_rate':
            [1e-4, 1e-3, 1e-2, 1e-1],
        'histgradientboostingclassifier__l2_regularization': [0, 1e-2, 1e-1],
        'histgradientboostingclassifier__max_depth': [None, 3]
        }

    # Define imputation models
    # ------------------------
    # Simple Imputer
    mean_imp = SimpleImputer(strategy='mean')

    # KNN Imputer
    knn_imp = KNNImputer(n_neighbors=15, weights='distance')

    # Iterative Imputer with Bayesian Ridge
    br_imp = IterativeImputer(estimator=BayesianRidge(), min_value=0)

    # Iterative Imputer with Random Forest
    est_rf = RandomForestRegressor()
    rf_imp = IterativeImputer(estimator=est_rf, min_value=0)

    # Define the imputation-model combinations to be tested
    # -----------------------------------------------------
    models = {'knn': knn, 'lr': lr, 'rf': rf, 'gb': gb}
    imputations = {'knn_imp': knn_imp, 'br_imp': br_imp, 'mean_imp': mean_imp,
                   'rf_imp': rf_imp}
    grids = {'knn': grid_knn, 'lr': grid_lr, 'rf': grid_rf, 'gb': grid_gb}

    return imputations, models, grids


def simbologia(volcano, event):

    simbología = pd.read_csv('/home/cmarmo/software/BOOM-master/Scripts/Simbologia.csv',
                             encoding='latin1', low_memory=False)
    Event = simbología.loc[simbología['Volcano'] == volcano]
    Event = Event.loc[Event['Event'] == event]
    coloR = Event.values[0, 2]
    markeR = Event.values[0, 3]
    return coloR, markeR


def colores(Y, type):
    Dpal = {}
    if type == 'volcano':
        for i, volcan in enumerate(np.unique(Y)):
            color, marker = simbologia(volcan, 'Unknown')
            Dpal[volcan] = color

    if type == 'event':
        simbología = pd.read_csv('/home/cmarmo/software/BOOM-master/Scripts/Simbologia.csv',
                                 encoding='latin1', low_memory=False)
        for event in np.unique(Y):
            # print(event)
            color, marker = simbologia(
                simbología[simbología.Event == event].Volcano.values[0], event)
            Dpal[event] = color

    return Dpal
