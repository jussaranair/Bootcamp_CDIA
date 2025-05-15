#### Projeto Final do Bootcamp CDIA ####
#### Centro Universitario SENAI/SC - Campus Florianopolis ####
#### Jussara Nair de Lima S. Andre ####

import streamlit as st
import pandas as pd
from utils.utils import pre_processing, save_pickle_to_file


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
                  'indice_de_luminosidade', 'sigmoide_das_areas', 'minimo_da_luminosidade',
                  'falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros'
                  ]
    if file_content.keys().to_list() != csv_header:
        st.warning('Arquivo fora do padrão!', icon=':material/warning:')
        return
    return pre_processing(file_content, is_train=True)


def main():
    st.title('Faça o upload do arquivo contendo os dados de treinamento!')
    file = st.file_uploader('Importar dados de treinamento.', type='csv', label_visibility='hidden')
    if file is not None:
        data = import_csv(file)
        if data is not None:
            if save_pickle_to_file(data, 'pre_processed_train'):
                st.info('Dados tratados!')

if __name__ == '__main__':
    main()
