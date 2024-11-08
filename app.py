import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Configuração da página
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Título
st.title("Aplicativo de Previsão de Preços de Imóveis")

# Upload do arquivo
uploaded_file = st.file_uploader("Faça o upload do arquivo CSV com os dados de imóveis", type="csv")

if uploaded_file:
    # Carregar dados
    house_data = pd.read_csv(uploaded_file)

    # Limpeza dos dados
    house_data = house_data.drop(columns=['date'], errors='ignore')

    # Visualização inicial dos dados
    st.write("Amostra dos dados:")
    st.write(house_data.head())

    # Visualização da matriz de correlação
    st.subheader("Matriz de Correlação")
    correlation_matrix = house_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribuições de variáveis principais
    st.subheader("Distribuições das Variáveis Principais")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(house_data['price'], bins=30, ax=axes[0, 0], kde=True).set_title('Distribuição de Preço')
    sns.histplot(house_data['sqft_living'], bins=30, ax=axes[0, 1], kde=True).set_title('Distribuição de Tamanho (sqft)')
    sns.histplot(house_data['bedrooms'], bins=15, ax=axes[1, 0], kde=True).set_title('Distribuição de Quartos')
    sns.histplot(house_data['bathrooms'], bins=15, ax=axes[1, 1], kde=True).set_title('Distribuição de Banheiros')
    st.pyplot(fig)

    # Seleção de features e target
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long']
    X = house_data[features]
    y = house_data['price']

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar e treinar modelos
    model_dict = {
        "Regressão Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    rmse_scores = {}
    st.subheader("Treinamento e Avaliação de Modelos")

    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores[model_name] = rmse
        st.write(f"{model_name} - RMSE: {rmse:.2f}")

    # Exibir RMSE dos modelos
    st.write("Comparação de RMSE dos Modelos:")
    st.write(pd.DataFrame(rmse_scores, index=["RMSE"]).T)

    # Normalizar dados e aplicar K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    house_data['cluster'] = kmeans.fit_predict(X_scaled)

    # Exibir clusters
    st.subheader("Clusters com K-Means")
    st.write(house_data[['price', 'cluster']].head())

    # Exibir gráfico de dispersão dos clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x=house_data['lat'], y=house_data['long'], hue=house_data['cluster'], palette='viridis', ax=ax)
    plt.title("Clusters por Localização")
    st.pyplot(fig)
else:
    st.info("Por favor, faça o upload do arquivo CSV.")
