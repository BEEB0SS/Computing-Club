from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('Life_Expectancy_Data.csv')

# Assuming 'data' has two columns for simplicity, which we'll use for clustering
X = data[['Year', 'Life_expectancy']]

# Implementing k-means clustering
kmeans = KMeans(n_clusters=3)
data['cluster'] = kmeans.fit_predict(X)

# Plotting clusters
plt.scatter(data['Year'], data['Life_expectancy'], c=data['cluster'], cmap='rainbow')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('K-means Clustering')
plt.show()
