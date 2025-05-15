#### Projeto Final do Bootcamp CDIA ####
#### Centro Universitario SENAI/SC - Campus Florianopolis ####
#### Jussara Nair de Lima S. Andre ####

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import pre_processing, pickle_to_models_dict, get_best_estimator_by_accuracy

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

def import_csv(file_name):
    file_content = pd.read_csv(file_name)
    csv_header = ['id', 'x_minimo', 'x_maximo', 'y_minimo', 'y_maximo', 'peso_da_placa',
                  'area_pixels', 'perimetro_x', 'perimetro_y', 'soma_da_luminosidade',
                  'maximo_da_luminosidade', 'comprimento_do_transportador',
                  'tipo_do_aço_A300', 'tipo_do_aço_A400', 'espessura_da_chapa_de_aço',
                  'temperatura', 'index_de_bordas', 'index_vazio', 'index_quadrado',
                  'index_externo_x', 'indice_de_bordas_x', 'indice_de_bordas_y',
                  'indice_de_variacao_x', 'indice_de_variacao_y', 'indice_global_externo',
                  'log_das_areas', 'log_indice_x', 'log_indice_y', 'indice_de_orientaçao',
                  'indice_de_luminosidade', 'sigmoide_das_areas', 'minimo_da_luminosidade'
                  ]
    if file_content.keys().to_list() != csv_header:
        st.warning('Arquivo fora do padrão!', icon=':material/warning:')
        return
    return file_content

def get_y_proba(X, model):
    X_test = pre_processing(X, is_train=False)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    proba_csv = pd.DataFrame(y_proba, columns=['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros'])
    proba_csv['id'] = X['id']
    proba_csv.to_csv('../content/proba.csv', columns=['id', 'falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros'], index=False)
    return y_pred, proba_csv

@st.cache_data(ttl='1h')
def get_json_from_api(token):
    headers = {"X-API-Key": token}
    files = {"file": open('../content/proba.csv', "rb")}
    params = {"threshold": 0.5}
    response = requests.post("http://3.138.252.216:5000/evaluate/multilabel_metrics", headers=headers, files=files, params=params)
    return response.json(), response.status_code

def main():
    st.title('Teste do melhor modelo treinado.')
    st.header('Faça o upload do arquivo contendo os dados de teste!')
    file = st.file_uploader('Importar dados de teste.', type='csv', label_visibility='hidden')
    if file is not None:
        data = import_csv(file)
        if data is not None:
            X = data.copy()
            models = pickle_to_models_dict('models')
            if models is None:
                st.warning('Modelos ainda não treinados. Retornando para Treinamento.', icon=':material/warning:')
                time.sleep(3)
                st.switch_page('pages/Treinamento.py')
            name, model = get_best_estimator_by_accuracy(models)
            y_pred, y_proba = get_y_proba(X, model['estimator'])
            threshold = st.slider('Threshold', 0.0, 1.0, 0.5)
            token = st.text_input('Insira o token')
            if token:
                if st.button('Realizar requisição na API'):
                    data, response = get_json_from_api(token)
                    if response != 200:
                        st.cache_data.clear()
                    else:
                        with st.expander("Metricas gerais", expanded=True):
                            col1 = st.columns(2)
                            with col1[0]:
                                st.write('Acuracia')
                                st.write(f'{data['macro_accuracy']*100:.2f}%')
                            with col1[1]:
                                st.write('ROC')
                                st.write(f'{data['macro_roc_auc']:.2f}')
                        with st.expander("Metricas por classe (Acuracia, Precisão, Recall, F1-Score)", expanded=True):
                            col2 = st.columns(7)
                            for i in range(0, 7):
                                with col2[i]:
                                    st.write(f'Classe {i+1}')
                                    st.write(f'{data['accuracy'][i]*100:.2f}%')
                                    st.write(f'{data['precision'][i]:.2f}')
                                    st.write(f'{data['recall'][i]:.2f}')
                                    st.write(f'{data['f1_score'][i]:.2f}')
                        with st.expander("Matriz de confusão", expanded=True):
                            col3 = st.columns(7)
                            for i in range(0, 7):
                                with col3[i]:
                                    st.write(f'Classe {i+1}')
                                    fig, ax = plt.subplots()
                                    sns.heatmap(data['confusion_matrix'][i], annot=True, fmt='d', cmap='Blues', ax=ax)
                                    st.pyplot(fig)
                    

if __name__ == '__main__':
    st.set_page_config(
    page_title="BootCamp",
    layout="wide",
    initial_sidebar_state="expanded",
    )
    main()
