import pandas as pd
import numpy as np

#exploration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def find_optimal_k(cluster_vars):
    """
    Uses kmeans and the elbow method to help find optimal k
    returns SSE and elbow plot of k
    """
    ks = range(2,20)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(cluster_vars)
        
        sse.append(kmeans.inertia_)
        
    print(pd.DataFrame(dict(k=ks, sse=sse)))
    
    
    plt.figure(figsize=(13,10))
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.grid()
    plt.show()

def elbow_k_means_plots_2_vars(num1, num2, features_list):
    """
    plots 2 variables with various range of K using kmeans clustering
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 16), sharex=True, sharey=True)

    for ax, k in zip(axs.ravel(), range(num1, num2)):
        clusters = KMeans(k).fit(features_list).predict(features_list)
        ax.scatter(features_list.iloc[:,0], features_list.iloc[:,1], c=clusters)
        ax.set(title='k = {}'.format(k))