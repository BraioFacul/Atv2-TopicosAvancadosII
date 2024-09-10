import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np  

# Carregar o dataset (substitua pela URL da base de dados se necessário)
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

# Calcular métricas de avaliação
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
dbscan_silhouette = silhouette_score(scaled_df, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

kmeans_db_score = davies_bouldin_score(scaled_df, kmeans_labels)
dbscan_db_score = davies_bouldin_score(scaled_df, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

# Mostrar métricas de avaliação
print(f'K-Means Silhouette Score: {kmeans_silhouette}')
print(f'K-Means Davies-Bouldin Score: {kmeans_db_score}')
print(f'DBSCAN Silhouette Score: {dbscan_silhouette}')
print(f'DBSCAN Davies-Bouldin Score: {dbscan_db_score}')

# Escolher o melhor algoritmo
if kmeans_silhouette > dbscan_silhouette:
    best_model = kmeans
    best_labels = kmeans_labels
    best_name = 'K-Means'
else:
    best_model = dbscan
    best_labels = dbscan_labels
    best_name = 'DBSCAN'

# Adicionar os rótulos de cluster ao dataframe original
df['Cluster'] = best_labels

# Mostrar o melhor algoritmo e os clusters
print(f'Melhor algoritmo: {best_name}')
print(f'Clusters: {set(best_labels)}')

# Plotar o gráfico de dispersão com os clusters
plt.figure(figsize=(12, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Ano')  
plt.ylabel('País')  
plt.title('Distribuição dos Clusters')
plt.colorbar(label='Cluster')
plt.show()

# Plotar gráficos dos clusters separados
plt.figure(figsize=(12, 6))
for cluster in set(best_labels):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], label=f'Cluster {cluster}')
plt.xlabel('Ano')  
plt.ylabel('País')  
plt.title('Clusters Encontrados')
plt.legend()
plt.show()

# Selecionar apenas colunas numéricas para as estatísticas
numeric_df = df.select_dtypes(include=[np.number])

# Gerar estatísticas descritivas (média, desvio padrão, mínimo e máximo) para cada cluster
cluster_stats = numeric_df.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])

# Mostrar estatísticas descritivas
print('Estatísticas descritivas dos clusters:')
print(cluster_stats)
