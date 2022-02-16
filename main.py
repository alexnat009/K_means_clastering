import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

np.random.seed(20)

# number of cluster was found by elbow method, with finding the
# optimal point where graph decrease drastically slows down
number_of_clusters = 9

df = pd.read_excel('Book1.xlsx', sheet_name='data')

km = KMeans(n_clusters=number_of_clusters, random_state=1)

y_predicted = km.fit_predict(df[['longitude', 'latitude']])
df['cluster'] = y_predicted

dfs = []
for i in range(number_of_clusters):
    tmp = df[df.cluster == i]
    dfs.append(tmp)

random_color = np.random.uniform(0.1, 0.9, (number_of_clusters, 3))

for k in range(number_of_clusters):
    plt.scatter(dfs[k]['longitude'], dfs[k]['latitude'],
                color=(random_color[k][0], random_color[k][1], random_color[k][2]))

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='*', label='centroid')

plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.show()
# The results are reproducible, all points remain fixed at their locations.
# It's just for the sake of beauty each region's color changes every time you compile code.
