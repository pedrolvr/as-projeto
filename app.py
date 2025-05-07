import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import riskfolio as rp
import statsmodels.api as sm
from streamlit_option_menu import option_menu
from curl_cffi import requests
import joblib

session = requests.Session(impersonate="chrome")

# Mapeamento de indicadores
indicator_mapping = {
    'cc': 'Control of Corruption',
    'ge': 'Government Effectiveness',
    'pv': 'Political Stability and Absence of Violence/Terrorism',
    'rl': 'Rule of Law',
    'rq': 'Regulatory Quality',
    'va': 'Voice and Accountability'
}

# Configuração dos modelos por índice
index_models = {
    'ibov': {'model': 'Log-Lin', 'variables': ['ge']},
    'nikkei': {'model': 'Linear-Linear', 'variables': ['pv', 'rl', 'rq']},
    'sti': {'model': 'Log-Lin', 'variables': ['ge', 'rl']},
    'asx200': {'model': 'Linear-Linear', 'variables': ['rl', 'rq']},
    'sp500': {'model': 'Log-Lin', 'variables': ['va']},
    'ssmi': {'model': 'Linear-Linear', 'variables': ['cc', 'ge', 'pv', 'va']},
    'mexbol': {'model': 'Linear-Linear', 'variables': ['cc', 'pv']},
    'dax': {'model': 'Log-Lin', 'variables': ['cc', 'rq']}
}

# Mapeamento de índice para país
index_to_country = {
    'ibov': 'Brasil',
    'sp500': 'Estados Unidos',
    'dax': 'Alemanha',
    'nikkei': 'Japão',
    'sti': 'Singapura',
    'asx200': 'Austrália',
    'mexbol': 'México',
    'ssmi': 'Suíça'
}

# Função de previsão
def predict_new_values(idx, x, values, model_name):
    # Validate model_name
    valid_models = ['Linear-Linear', 'Log-Lin', 'Log-Log']
    if model_name not in valid_models:
        print(f"Error: model_name must be one of {valid_models}")
        return None

    # Create DataFrame from the provided values (simulating what would come from the CSV)
    df = pd.DataFrame([values], columns=x)

    # Prepare model data based on model_name
    if model_name == 'Linear-Linear':
        X_var = df
    elif model_name == 'Log-Lin':
        X_var = np.log(df)
    elif model_name == 'Log-Log':
        X_var = np.log(df)
    
    # Check for invalid values
    if np.any(np.isnan(X_var)) or np.any(np.isinf(X_var)):
        print(f"Error: Invalid values (NaN or inf) in features after transformation for {model_name}")
        return None

    # Add constant for statsmodels
    X_var_const = sm.add_constant(X_var, has_constant='add')

    # Load the saved model
    model_file = f'models/{idx}/{idx}_{model_name}.pkl'
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found")
        return None

    model = sm.load(model_file)

    # Load the saved scaler
    scaler_file = f'models/{idx}/{idx}_scaler_y.pkl'
    if not os.path.exists(scaler_file):
        print(f"Error: Scaler file {scaler_file} not found")
        return None

    scaler_y = joblib.load(scaler_file)

    # Predict using the model
    try:
        pred_norm = model.predict(X_var_const)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
    # Inverse-transform predictions
    if model_name == 'Log-Log':
        # Exponentiate to reverse log transformation
        pred_norm = np.exp(pred_norm)

    # Inverse-transform to original scale
    pred_orig = scaler_y.inverse_transform(pred_norm.values.reshape(-1, 1)).ravel()

    # Create a DataFrame with predicted values
    results = pd.DataFrame({
        f'Predicted {idx.upper()}': pred_orig
    })

    # Display the table
    print(f"\n{idx.upper()} Predictions (Using {model_name} Model):")
    print(results.to_string(index=False))

    return results

# Menu lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Navegação",
        options=["Sobre", "Previsão"],
        icons=["info-circle", "bar-chart", "graph-up"],
        default_index=0,
    )

# Página 'Sobre'
if selected == 'Sobre':
    st.title("Sobre a Aplicação")
    st.markdown("""
    O **App de Previsão de Índices de Mercado** é uma ferramenta avançada projetada para prever o desempenho de índices financeiros globais. Abaixo, você encontrará um resumo das principais funcionalidades e características da aplicação:

    ### 1. **Índices de Mercado Previstos**
    O aplicativo faz previsões para diversos índices financeiros, incluindo:
    - IBOV
    - NIKKEI
    - STI
    - ASX200
    - SP500
    - SSMI
    - MEXBOL
    - DAX

    ### 2. **Modelos Estatísticos Utilizados**
    A previsão dos índices é realizada através de modelos de **regressão log-linear e linear**. Variáveis econômicas como Efetividade Governamental, Estado de Direito e Qualidade Regulatória são analisadas para prever o comportamento futuro desses índices.

    ### 3. **Método de Estimação**
    O app utiliza o método de **Ordinary Least Squares (OLS)** para ajustar os modelos de regressão, garantindo uma análise estatística robusta.

    ### 4. **Métricas de Desempenho**
    O aplicativo oferece uma análise detalhada do desempenho dos modelos, com as seguintes métricas:
    - **R² ajustado**: Medida de como as variáveis explicativas explicam a variação do índice.
    - **Erro Médio Absoluto (MAE)**: Indica a precisão das previsões.
    
    Além disso, são realizados testes de **normalidade dos resíduos** e **avaliação de multicolinearidade** para assegurar a qualidade do modelo.

    ### 5. **Como Usar**
    Para utilizar o app, siga os passos abaixo:
    1. **Seleção do Índice**: Escolha o índice de mercado que deseja prever.
    2. **Configuração das Variáveis**: Ajuste as variáveis econômicas que influenciam a previsão do índice.
    3. **Execução da Previsão**: Clique no botão de previsão para obter os resultados.
    """)


# Página 'Previsão'
elif selected == 'Previsão':
    st.title("Previsão de Índices")

    # Seleciona índice
    options = {f"{idx.upper()} ({index_to_country[idx]})": idx for idx in index_models.keys()}
    selected_label = st.selectbox("Escolha o índice:", list(options.keys()))
    selected_idx = options[selected_label]

    # Pega informações do modelo do índice selecionado
    model_info = index_models[selected_idx]
    model_name = model_info['model']
    variables = model_info['variables']

    st.markdown("### Insira os valores para os indicadores:")
    input_values = []
    for var in variables:
        val = st.number_input(f"{indicator_mapping[var]} ({var})", min_value=0.01, value=1.0)
        input_values.append(val)

    if st.button("Calcular Previsão"):
        results = predict_new_values(selected_idx, variables, input_values, model_name)
        if results is not None:
            st.table(results)
