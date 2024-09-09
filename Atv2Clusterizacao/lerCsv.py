import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np  # Importação do numpy necessária

# Carregar o dataset
df = pd.read_csv('Atv2Clusterizacao/climate_change_impact_on_agriculture_2024.csv')

# Pré-processamento (remover NaN, normalizar variáveis)
df = df.dropna()
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_df)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_df)

# Escolher o melhor algoritmo
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
dbscan_silhouette = silhouette_score(scaled_df, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

if kmeans_silhouette > dbscan_silhouette:
    best_model = kmeans
    best_labels = kmeans_labels
    best_name = 'K-Means'
else:
    best_model = dbscan
    best_labels = dbscan_labels
    best_name = 'DBSCAN'

# Adicionar coluna de clusters ao dataframe
df['Cluster'] = best_labels

# Gráfico 1: Distribuição de clusters usando K-Means ou DBSCAN
plt.figure(figsize=(12, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Feature 1')  # Altere conforme as suas colunas
plt.ylabel('Feature 2')  # Altere conforme as suas colunas
plt.title('Distribuição dos Clusters')
plt.colorbar(label='Cluster')
plt.show()

# Gráfico 2: Gráfico de dispersão com cores diferentes para cada cluster
plt.figure(figsize=(12, 6))
for cluster in set(best_labels):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {cluster}')
plt.xlabel('Feature 1')  # Altere conforme as suas colunas
plt.ylabel('Feature 2')  # Altere conforme as suas colunas
plt.title('Clusters Encontrados')
plt.legend()
plt.show()
