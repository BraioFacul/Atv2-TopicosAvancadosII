import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('C:/Users/BRYAN/PycharmProjects/Atv2Clusterizacao/climate_change_impact_on_agriculture_2024.csv')

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

# Calcular métricas de avaliação
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
kmeans_db_score = davies_bouldin_score(scaled_df, kmeans_labels)
dbscan_silhouette = silhouette_score(scaled_df, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
dbscan_db_score = davies_bouldin_score(scaled_df, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

# Escolher o melhor algoritmo
if kmeans_silhouette > dbscan_silhouette:
    best_model = kmeans
    best_labels = kmeans_labels
    best_name = 'K-Means'
else:
    best_model = dbscan
    best_labels = dbscan_labels
    best_name = 'DBSCAN'

# Aplicar o melhor modelo novamente
df['Cluster'] = best_labels

# Selecionar apenas colunas numéricas para as estatísticas
numeric_df = df.select_dtypes(include=[np.number])
cluster_stats = numeric_df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])

# Mostrar resultados
print(f'Melhor algoritmo: {best_name}')
print(f'Silhouette Score: {silhouette_score(scaled_df, best_labels)}')
print(f'Davies-Bouldin Score: {davies_bouldin_score(scaled_df, best_labels)}')
print(cluster_stats)