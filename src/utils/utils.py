#### Projeto Final do Bootcamp CDIA ####
#### Centro Universitario SENAI/SC - Campus Florianopolis ####
#### Jussara Nair de Lima S. Andre ####

import os
import pickle
import pandas as pd
from scipy import stats

def fill_null(df):
    # Preenchimento de dados nulos
    df = df.fillna(df.mean(numeric_only=True))
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_columns(df):
    # Remocao de colunas ociosas
    return df.drop(columns=['id', 'peso_da_placa'], axis=1)

def abs_columns(df):
    # Remocao de valores negativos improprios
    df['area_pixels'] = df['area_pixels'].abs()
    df['perimetro_x'] = df['perimetro_x'].abs()
    df['perimetro_y'] = df['perimetro_y'].abs()
    df['comprimento_do_transportador'] = df['comprimento_do_transportador'].abs()
    df['espessura_da_chapa_de_aço'] = df['espessura_da_chapa_de_aço'].abs()
    return df

def a300_to_category(df):
    # Padronizacao do atributo "Tipo_do_aço_A300"
     df['tipo_do_aço_A300'] = df['tipo_do_aço_A300'].map({
        '-': '0',
        '0': '0',
        'N': '0',
        'Não': '0',
        'não': '0',
        'Sim': '1',
        'sim': '1',
        '1': '1'
        })
     return df

def a400_to_category(df):
    # Padronizacao do atributo "Tipo_do_aço_A400"
    df['tipo_do_aço_A400'] = df['tipo_do_aço_A400'].map({
        '0': '0',
        'Não': '0',
        'não': '0',
        'nao': '0',
        'Sim': '1',
        'sim': '1',
        'S': '1',
        '1': '1'
        })
    return df

def falha_1_to_category(df):
    # Padronizacao e conversao para booleano dos dados da "falha_1"
    df['falha_1'] = df['falha_1'].map({
        '0': False,
        'nao': False,
        'False': False,
        'S': True,
        '1': True,
        'True': True
        })
    return df

def falha_2_to_category(df):
    # Padronizacao e conversao para booleano dos dados da "falha_2"
    df['falha_2'] = df['falha_2'].map({
        '0': False,
        'False': False,
        'S': True,
        'y': True,
        '1': True,
        'True': True
        })
    return df

def falha_4_to_category(df):
    # Padronizacao e conversao para booleano dos dados da "falha_4"
    df['falha_4'] = df['falha_4'].map({
        '0': False,
        'nao': False,
        'False': False,
        'S': True,
        '1': True,
        'True': True
        })
    return df

def falha_5_to_category(df):
    # Padronizacao e conversao para booleano dos dados da "falha_5"
    df['falha_5'] = df['falha_5'].map({
        'não': False,
        'Não': False,
        'Sim': True,
        'sim': True
        })
    return df

def falha_outros_to_category(df):
    # Padronizacao e conversao para booleano dos dados da "falha_outros"
    df['falha_outros'] = df['falha_outros'].map({
        'Não': False,
        'Sim': True,
        })
    return df

def remove_outliers(df):
    # Remocao dos outliers com z_scores
    for col in df.select_dtypes(include='number').columns:
        z_scores = stats.zscore(df[col])
        df = df[(z_scores < 5) & (z_scores > -5)]
    return df

def pre_processing(df, is_train=False):
    df = fill_null(df)
    df = drop_columns(df)
    df = abs_columns(df)
    if is_train:
        df = a300_to_category(df)
        df = a400_to_category(df)
        df = falha_1_to_category(df)
        df = falha_2_to_category(df)
        df = falha_4_to_category(df)
        df = falha_5_to_category(df)
        df = falha_outros_to_category(df)
        df = remove_outliers(df)
    return df

def make_file_path(name):
    return os.path.join('../content/' + name + '.pkl')

def check_if_file_exists(file_path):
    if os.path.exists(file_path):
        return True
    return False

def save_pickle_to_file(df, name):
    file_path = make_file_path(name)
    df.to_pickle(file_path)
    return check_if_file_exists(file_path)

def get_pickle_from_file(name):
    file_path = make_file_path(name)
    if check_if_file_exists(file_path):
        return pd.read_pickle(file_path)
    return

def models_dict_to_pickle(models, name):
    file_path = make_file_path(name)
    with open(file_path, 'wb') as file:
        pickle.dump(models, file)
    return check_if_file_exists(file_path)

def pickle_to_models_dict(name):
    file_path = make_file_path(name)
    if check_if_file_exists(file_path):
        with open(file_path, 'rb') as file:
            models = pickle.load(file)
        return models
    return

def get_best_estimator_by_accuracy(models):
    best = None
    best_score = 0
    for model in models:
        if models[model]['score'] > best_score:
            best_score = models[model]['score']
            best = model
    return best, models[best]
