import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
import riskfolio as rp
from streamlit_option_menu import option_menu

index_to_country = {
    'IBOV': 'Brasil',
    'FTSE100': 'Reino Unido',
    'SP500': 'Estados Unidos',
    'DAX': 'Alemanha',
    'NIKKEI': 'Japão',
    'HSI': 'Hong Kong',
    'STI': 'Singapura',
    'ASX200': 'Austrália',
    'KOSPI': 'Coreia do Sul',
    'SENSEX': 'Índia',
    'MEXBOL': 'México',
    'IBEX': 'Espanha',
    'SSMI': 'Suíça'
}

def download_data(start_date, end_date):
    indices = {
        'IBOV': '^BVSP', 'FTSE100': '^FTSE', 'SP500': '^GSPC', 'DAX': '^GDAXI', 'NIKKEI': '^N225',
        'HSI': '^HSI', 'STI': '^STI', 'ASX200': '^AXJO', 'KOSPI': '^KS11', 'SENSEX': '^BSESN',
        'MEXBOL': '^MXX', 'IBEX': '^IBEX', 'SSMI': '^SSMI'
    }
    data = {}
    for name, ticker in indices.items():
        data[name] = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)['Close']
    return data

def calculate_returns(data):
    returns = pd.DataFrame({name: df.resample('M').ffill().pct_change() for name, df in data.items()})
    return returns.dropna()

def filter_indices_by_indicator(indicator, threshold, indicators):
    return [index for index, values in indicators.items() if values[indicator] >= threshold]

def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)

def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def objective(weights, mean_returns, cov_matrix, target_return):
    return portfolio_risk(weights, cov_matrix) - target_return * portfolio_return(weights, mean_returns)

def optimal_portfolio_markowitz(mean_returns, cov_matrix, target_return):
    initial_guess = [1/len(mean_returns)] * len(mean_returns)
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(objective, initial_guess, args=(mean_returns, cov_matrix, target_return),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimal_portfolio_hrp(returns):
    port = rp.HCPortfolio(returns=returns)
    weights = port.optimization(model='HRP', rm='MV', rf=0.0)
    return weights.values.flatten()

with st.sidebar:
    selected = option_menu(
        menu_title="Navegação",
        options=["Sobre", "Carteira"],
        icons=["info-circle", "bar-chart"],
        default_index=0,
    )

if selected == 'Sobre':
    st.title("Sobre a Aplicação")
    st.markdown("""
    Esta aplicação realiza a **otimização de carteiras de investimento** com base em índices de mercado globais, utilizando dois modelos: **Markowitz** (otimização de média-variância) e **Hierarchical Risk Parity (HRP)**. O objetivo é ajudar investidores a construir carteiras diversificadas, considerando desempenho financeiro e indicadores de governança.

    ### Modelos Disponíveis:
    - **Markowitz**:
    - Usa o modelo clássico de média-variância para minimizar o risco (volatilidade) para um retorno alvo definido pelo usuário.
    - Otimiza os pesos dos ativos usando o método SLSQP, sujeito a restrições (soma dos pesos = 1, pesos entre 0 e 1).
    - Requer a especificação de um retorno alvo mensal.
    - **Hierarchical Risk Parity (HRP)**:
    - Aloca pesos com base em clustering hierárquico da matriz de correlação dos ativos, distribuindo o risco de forma mais equilibrada.
    - Não requer um retorno alvo, tornando-o mais robusto a estimativas imprecisas de retornos.
    - Usa a biblioteca `riskfolio-lib` para implementação eficiente.

    ### Como Funciona a Seleção de Índices:
    A aplicação utiliza os **indicadores do World Governance Indicators (WGI)** para gerar uma **análise de regressão múltipla** para cada índice de mercado global. Com base nos coeficientes da regressão, a aplicação filtra os índices a serem utilizados na construção da carteira. Ou seja, índices que atendem aos critérios definidos pelos indicadores de governança (como **Controle da Corrupção**, **Estabilidade Política**, **Eficácia do Governo**, etc.) são selecionados para a otimização da carteira.

    - **WGI (World Governance Indicators)**: Os indicadores incluem fatores como **Controle da Corrupção**, **Estabilidade Política**, **Eficácia do Governo**, **Qualidade Regulatória**, **Estado de Direito**, e **Voz e Responsabilidade**.
    - A análise de **regressão múltipla** é aplicada para entender como cada índice de mercado está relacionado com os indicadores de governança, e os coeficientes da regressão são **normalizados entre -100 e 100** para garantir uma escala comum entre os diferentes índices. Esses coeficientes normalizados são então usados para **filtrar** os índices de acordo com o valor mínimo do indicador escolhido pelo usuário.

    ### Funcionalidades:
    - **Download de dados**: Obtém preços históricos de índices como IBOV, S&P 500, FTSE 100, entre outros, via `yfinance`.
    - **Cálculo de retornos**: Calcula retornos mensais dos índices com base nos preços de fechamento.
    - **Filtro por indicadores de governança**: Filtra índices com base em indicadores como Controle da Corrupção, Estabilidade Política, Eficácia do Governo, entre outros (valores simulados).
    - **Otimização da carteira**: Permite escolher entre Markowitz (com retorno alvo) ou HRP (alocação baseada em risco).
    - **Interface interativa**: Interface amigável no Streamlit, onde o usuário pode:
    - Definir o período de análise (data de início e fim).
    - Escolher o modelo de otimização (Markowitz ou HRP).
    - Para Markowitz, especificar o retorno alvo mensal.
    - Selecionar um indicador de governança e seu valor mínimo.
    - **Resultados**: Exibe os índices selecionados e suas alocações em uma tabela, além do risco mensal, risco anualizado e retorno estimado.

    ### Como usar:
    1. Navegue até a página "Carteira" clicando no botão no menu lateral.
    2. Escolha o modelo de otimização (Markowitz ou HRP).
    3. Defina o período de análise e o indicador de governança.
    4. Para Markowitz, ajuste o retorno alvo; para HRP, o retorno alvo é ignorado.
    5. Ajuste o valor mínimo do indicador com o slider.
    6. Clique em "Calcular Carteira" para ver os resultados em uma tabela.

    """)

elif selected == 'Carteira':
    st.title("Análise de Carteira de Investimentos")

    # Inputs
    start_date = st.date_input("Data de Início", pd.to_datetime("1996-01-01"))
    end_date = st.date_input("Data de Fim", pd.to_datetime("2023-12-31"))
    model = st.selectbox("Modelo de Otimização", ["Markowitz", "HRP"])
    target_return = st.slider("Retorno Alvo da Carteira (mensal, usado apenas no Markowitz)", 0.01, 0.30, 0.15, 0.01)

    indicator_map = {
        'Voz e Responsabilidade': 'va',
        'Estabilidade Política e Ausência de Violência/Terrorismo': 'pv',
        'Eficácia do Governo': 'ge',
        'Qualidade Regulatória': 'rq',
        'Estado de Direito': 'rl',
        'Controle da Corrupção': 'cc'
    }

    indicator_name = st.selectbox("Escolha o indicador", list(indicator_map.keys()))
    indicator = indicator_map[indicator_name]  # Converte para o código técnico
    threshold = st.slider(f"Valor mínimo para o indicador {indicator_name}", -100.0, 100.0, 0.0, 1.0)

    indicators = {
        'IBOV': {'cc': 100.00, 'ge': -79.75, 'pv': 41.02, 'rl': 87.66, 'rq': -100.00, 'va': -1.55},
        'FTSE100': {'cc': 57.14, 'ge': -100.00, 'pv': 100.00, 'rl': -16.73, 'rq': 66.02, 'va': 94.94},
        'SP500': {'cc': 94.74, 'ge': 36.38, 'pv': 60.97, 'rl': 36.15, 'rq': 100.00, 'va': -100.00},
        'DAX': {'cc': 80.67, 'ge': -100.00, 'pv': 3.77, 'rl': -62.66, 'rq': 100.00, 'va': 8.85},
        'NIKKEI': {'cc': -100.00, 'ge': 43.72, 'pv': 64.28, 'rl': 100.00, 'rq': 97.09, 'va': 32.46},
        'STI': {'cc': -90.54, 'ge': 61.10, 'pv': 61.53, 'rl': 100.00, 'rq': -100.00, 'va': -48.56},
        'ASX200': {'cc': 20.54, 'ge': 11.28, 'pv': -4.83, 'rl': -100.00, 'rq': 100.00, 'va': 0.01},
        'MEXBOL': {'cc': -100.00, 'ge': 17.95, 'pv': -60.15, 'rl': 100.00, 'rq': -25.99, 'va': 67.12},
        'IBEX': {'cc': -39.94, 'ge': -100.00, 'pv': -62.90, 'rl': -10.60, 'rq': 100.00, 'va': -60.68},
        'SSMI': {'cc': 75.78, 'ge': 100.00, 'pv': -89.03, 'rl': -5.83, 'rq': 9.76, 'va': -100.00}
    }


    if st.button("Calcular Carteira"):
        with st.spinner("Calculando a carteira ótima..."):
            data = download_data(start_date, end_date)
            returns = calculate_returns(data)
            filtered_indices = filter_indices_by_indicator(indicator, threshold, indicators)

            if filtered_indices:
                filtered_returns = returns[filtered_indices]
                mean_returns = filtered_returns.mean()
                cov_matrix = filtered_returns.cov()

                if model == "Markowitz":
                    weights = optimal_portfolio_markowitz(mean_returns, cov_matrix, target_return)
                    risk = portfolio_risk(weights, cov_matrix)
                    portfolio_ret = portfolio_return(weights, mean_returns)
                else:  # HRP
                    weights = optimal_portfolio_hrp(filtered_returns)
                    risk = portfolio_risk(weights, cov_matrix)
                    portfolio_ret = portfolio_return(weights, mean_returns)

                allocation_data = {
                    'Índice': filtered_indices,
                    'País': [index_to_country[index] for index in filtered_indices],
                    'Peso (%)': [weight * 100 for weight in weights]
                }
                allocation_df = pd.DataFrame(allocation_data)
                allocation_df = allocation_df[allocation_df['Peso (%)'] > 0].round(2)

                st.success("Carteira calculada com sucesso!")
                st.subheader("Alocação da Carteira")
                st.dataframe(allocation_df, use_container_width=True)

                st.subheader("Métricas da Carteira")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risco Mensal", f"{risk:.4f}")
                with col2:
                    st.metric("Risco Anualizado", f"{risk * np.sqrt(12):.4f}")
                with col3:
                    st.metric("Retorno Estimado Mensal", f"{portfolio_ret*100:.2f}%")
            else:
                st.warning("Nenhum índice atende ao critério de filtro.")