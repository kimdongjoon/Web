import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris_df = pd.read_csv('data/iris.csv')
del iris_df['Id']
target = LabelEncoder().fit_transform(iris_df['Species'])
del iris_df['Species']
iris_data = iris_df.values

ncls = 4
kmeans = KMeans(n_clusters=ncls, init='k-means++', max_iter=300,
                random_state=0)
kmeans.fit(iris_data)

iris_df['target'] = target
iris_df['cluster'] = kmeans.labels_

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris_data)

iris_df['pca_x'] = pca_transformed[:,0]
iris_df['pca_y'] = pca_transformed[:,1]

markers=['^', 's', 'o', '*', 'P', 'p']
iris_target_name = ['Setosa', 'Versicolor', 'Virginica']

# PCA 산점도
fig, ax = plt.subplots(figsize=(6,4))
for i in range(3):
    x_axis_data = iris_df[iris_df['target']==i]['pca_x']
    y_axis_data = iris_df[iris_df['target']==i]['pca_y']
    ax.scatter(x_axis_data, y_axis_data, marker=markers[i],
                label=iris_target_name[i])
ax.legend()
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
fig.savefig('static/images/pca.png')

# K-Means Clustering 산점도
fig, ax = plt.subplots(figsize=(6,4))
for i in range(ncls):
    x_axis_data = iris_df[iris_df['cluster']==i]['pca_x']
    y_axis_data = iris_df[iris_df['cluster']==i]['pca_y']
    ax.scatter(x_axis_data, y_axis_data, marker=markers[i])
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
fig.savefig('static/images/kmc.png')