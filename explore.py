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
        
def get_location_clusters(train, test):
    """
    Takes in train and test and uses latitude and longitutde to cluster into a 2 different location clusters
    one using k of 5 and the other k of 6
    returns train and test with columns for each in numeric and string form
    """

    # K of 5
    X_train = train[["latitude", "longitude"]]
    X_test = test[["latitude", "longitude"]]
    kmeans = KMeans(5)
    kmeans.fit(X_train)
    train["n_location_cluster_k5"] = kmeans.predict(X_train)
    train["s_location_cluster_k5"] = 'cluster_' + (train.n_location_cluster_k5).astype(str)
    test["n_location_cluster_k5"] = kmeans.predict(X_test)
    test["s_location_cluster_k5"] = 'cluster_' + (test.n_location_cluster_k5).astype(str)


    #K of 6
    X_train = train[["latitude", "longitude"]]
    X_test = test[["latitude", "longitude"]]
    kmeans = KMeans(6)
    kmeans.fit(X_train)
    train["n_location_cluster_k6"] = kmeans.predict(X_train)
    train["s_location_cluster_k6"] = 'cluster_' + (train.n_location_cluster_k6).astype(str)
    test["n_location_cluster_k6"] = kmeans.predict(X_test)
    test["s_location_cluster_k6"] = 'cluster_' + (test.n_location_cluster_k6).astype(str)
    
    return train, test
        
    
def t_test_location_cluster_k5(train):
    """Runs a T-Test for each cluster group in k5 against overall logerror mean
    alpha is set to a 95% confidence
    returns a p value for each cluster"""
    
    alpha = .05
    
    loc_0 = train[train.n_location_cluster_k5 == 0]
    t0, p0 = stats.ttest_1samp(loc_0.logerror, train.logerror.mean())
    if p0 < alpha:
        print("We reject the null hypothesis for Cluster 0")
    else:
        print("We fail to reject the null hypothesis for for Cluster 0")

    
    loc_1 = train[train.n_location_cluster_k5 == 1]
    t1, p1 = stats.ttest_1samp(loc_1.logerror, train.logerror.mean())
    if p1 < alpha:
        print("We reject the null hypothesis for Cluster 1")
    else:
        print("We fail to reject the null hypothesis for Cluster 1")
   
    loc_2 = train[train.n_location_cluster_k5 == 2]
    t2, p2 = stats.ttest_1samp(loc_2.logerror, train.logerror.mean())
    if p2 < alpha:
        print("We reject the null hypothesis for Cluster 2")
    else:
        print("We fail to reject the null hypothesis for Cluster 2")
        
    loc_3 = train[train.n_location_cluster_k5 == 3]
    t3, p3 = stats.ttest_1samp(loc_3.logerror, train.logerror.mean())
    if p3 < alpha:
        print("We reject the null hypothesis for Cluster 3")
    else:
        print("We fail to reject the null hypothesis for Cluster 3") 
        
    loc_4 = train[train.n_location_cluster_k5 == 4]
    t4, p4 = stats.ttest_1samp(loc_4.logerror, train.logerror.mean())
    if p4 < alpha:
        print("We reject the null hypothesis for Cluster 4")
    else:
        print("We fail to reject the null hypothesis for Cluster 4")
    
    print(f"Our p value for Cluster 0 is {p0:.2}")
    print(f"Our p value for Cluster 1 is {p1:.2}")
    print(f"Our p value for Cluster 2 is {p2:.2}")
    print(f"Our p value for Cluster 3 is {p3:.2}")
    print(f"Our p value for Cluster 4 is {p4:.2}")
    return p0, p1, p2, p3, p4



def t_test_location_cluster_k6(train):
    """Runs a T-Test for each cluster group in k6 against overall logerror mean
    alpha is set to a 95% confidence
    returns a p value for each cluster"""
    alpha = .05
    
    loc_0 = train[train.n_location_cluster_k6 == 0]
    t0, p0 = stats.ttest_1samp(loc_0.logerror, train.logerror.mean())
    if p0 < alpha:
        print("We reject the null hypothesis for Cluster 0")
    else:
        print("We fail to reject the null hypothesis for for Cluster 0")

    
    loc_1 = train[train.n_location_cluster_k6 == 1]
    t1, p1 = stats.ttest_1samp(loc_1.logerror, train.logerror.mean())
    if p1 < alpha:
        print("We reject the null hypothesis for Cluster 1")
    else:
        print("We fail to reject the null hypothesis for Cluster 1")
   
    loc_2 = train[train.n_location_cluster_k6 == 2]
    t2, p2 = stats.ttest_1samp(loc_2.logerror, train.logerror.mean())
    if p2 < alpha:
        print("We reject the null hypothesis for Cluster 2")
    else:
        print("We fail to reject the null hypothesis for Cluster 2")
        
    loc_3 = train[train.n_location_cluster_k6 == 3]
    t3, p3 = stats.ttest_1samp(loc_3.logerror, train.logerror.mean())
    if p3 < alpha:
        print("We reject the null hypothesis for Cluster 3")
    else:
        print("We fail to reject the null hypothesis for Cluster 3") 
        
    loc_4 = train[train.n_location_cluster_k6 == 4]
    t4, p4 = stats.ttest_1samp(loc_4.logerror, train.logerror.mean())
    if p4 < alpha:
        print("We reject the null hypothesis for Cluster 4")
    else:
        print("We fail to reject the null hypothesis for Cluster 4")
        
    loc_5 = train[train.n_location_cluster_k6 == 5]
    t5, p5 = stats.ttest_1samp(loc_4.logerror, train.logerror.mean())
    if p5 < alpha:
        print("We reject the null hypothesis for Cluster 5")
    else:
        print("We fail to reject the null hypothesis for Cluster 5")
    
    print(f"Our p value for Cluster 0 is {p0:.2}")
    print(f"Our p value for Cluster 1 is {p1:.2}")
    print(f"Our p value for Cluster 2 is {p2:.2}")
    print(f"Our p value for Cluster 3 is {p3:.2}")
    print(f"Our p value for Cluster 4 is {p4:.2}")
    print(f"Our p value for Cluster 5 is {p5:.2}")
    return p0, p1, p2, p3, p4, p5