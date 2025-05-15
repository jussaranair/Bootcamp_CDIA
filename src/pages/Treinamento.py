#### Projeto Final do Bootcamp CDIA ####
#### Centro Universitario SENAI/SC - Campus Florianopolis ####
#### Jussara Nair de Lima S. Andre ####

import streamlit as st
import pandas as pd
import numpy as np
import time

from utils.utils import get_pickle_from_file, models_dict_to_pickle, pickle_to_models_dict, get_best_estimator_by_accuracy

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector as SFS, SelectKBest, f_classif
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def split_x_y_from_data(data):
    y = data['falha_1'] + data['falha_2']*2 + data['falha_3']*3 + data['falha_4']*4 + data['falha_5']*5 + data['falha_6']*6

    X = data.drop(columns=['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros'], axis=1)
    return X, y

def get_cv(n_splits):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def ppline_search(X, y, cv):
    # Estimativa dos melhores parametros pelo randomized search
    # dentro de um pipeline pre-definido.

    models_params = [
        ('LogisticRegression', LogisticRegression(max_iter=1000), {
            'clf__C': [0.1, 1.0, 10],
            'clf__solver': ['lbfgs', 'liblinear']
        }),
        ('SVC', SVC(), {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'rbf']
        }),
        ('KNeighbors', KNeighborsClassifier(), {
            'clf__n_neighbors': [3, 5, 7],
            'clf__weights': ['uniform', 'distance']
        }),
        ('DecisionTree', DecisionTreeClassifier(), {
            'clf__max_depth': [3, 5, None],
            'clf__criterion': ['gini', 'entropy']
        }),
        ('RandomForest', RandomForestClassifier(), {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [None, 5],
        }),
        ('GradientBoosting', GradientBoostingClassifier(), {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.05, 0.1],
        }),
        ('AdaBoost', AdaBoostClassifier(), {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.5, 1.0]
        }),
        ('GaussianNB', GaussianNB(), {}),
        ('LDA', LinearDiscriminantAnalysis(), {
            'clf__solver': ['svd', 'lsqr']
        }),
        ('QDA', QuadraticDiscriminantAnalysis(), {
            'clf__reg_param': [0.0, 0.1, 0.5]
        }),
    ]

    models = {}
    for name, clf, params in models_params:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            #('feature_selection', SelectKBest(score_func=f_classif, k=2)),
            ('clf', clf)
        ])

        grid = RandomizedSearchCV(pipe, params, n_iter=10, cv=cv, n_jobs=-1, scoring='accuracy', verbose=10)
        grid.fit(X, y)

        score = cross_val_score(grid.best_estimator_, X, y, cv=cv, scoring='accuracy')

        models[name] = {'estimator': grid.best_estimator_, 'params': grid.best_params_, 'score': np.mean(score)}
    return models

def cross_val_classification_report(X, y, cv, model):
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reported_dict = classification_report(y_test, y_pred, output_dict=True)
        reported_dict = pd.DataFrame(reported_dict).transpose()

        st.header(f'{fold}º Fold')
        st.dataframe(reported_dict.style.format("{:.2f}"))

#def get_best_estimator_by_accuracy(models):
#    best = None
#    best_score = 0
#    for model in models:
#        if models[model]['score'] > best_score:
#            best_score = models[model]['score']
#            best = model
#    return best, models[best]

def main():
    st.title('Treinamento e validação dos modelos.')
    data = get_pickle_from_file('pre_processed_train')
    if data is None:
        st.warning('Dados ainda não tratados. Retornando para Home.', icon=':material/warning:')
        time.sleep(3)
        st.switch_page('Home.py')

    with st.sidebar:
        # TODO
        # n_splits = st.slider('Número de Folds', 3, 10, 3)
        n_splits = 3
    cv = get_cv(n_splits)
    X, y = split_x_y_from_data(data)
    models = pickle_to_models_dict('models')
    if models is None:
        with st.sidebar:
            run_ppline_search = st.button('Treinar modelos')
        if run_ppline_search:
            models = ppline_search(X, y, n_splits)
            models_dict_to_pickle(models, 'models')
            time.sleep(3)
            st.rerun()
    else:
        name, model = get_best_estimator_by_accuracy(models)
        st.write(f'O melhor modelo encontrado usando {n_splits} Folds foi o {name}. A acuracia media foi de {model['score']:.2f}')

if __name__ == '__main__':
    st.set_page_config(
    page_title="BootCamp",
    layout="wide",
    initial_sidebar_state="expanded",
    )
    main()
