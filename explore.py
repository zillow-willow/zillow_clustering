import pandas as pd
import numpy as np

#exploration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats



#Functions to visualize optimizing for K

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

   


#Functions for clustering
        
def cluster_target_variable(train, test):
    """
    Takes in train and test dataframe and clusters the log error with k 6
    returns train, test, kmeans
    """

    X_train = train[["logerror"]]
    X_test = test[["logerror"]]

    kmeans = KMeans(n_clusters=6, random_state = 123)
    train["n_cluster_target"] = kmeans.fit_predict(X_train)
    test["n_cluster_target"] = kmeans.predict(X_test)
    
    train["s_cluster_target"] = "Cluster " + train["n_cluster_target"].astype(str)
    test["s_cluster_target"] = "Cluster " + test["n_cluster_target"].astype(str)
    
    return train, test, kmeans
                
        
def get_location_clusters(train, test):
    """
    Takes in train and test and uses latitude and longitutde to cluster into a 2 different location clusters
    one using k of 5 and the other k of 6
    returns train ,test kmean5, kmeans 6
    """

    # K of 5
    X_train = train[["latitude", "longitude"]]
    X_test = test[["latitude", "longitude"]]
    kmeans5 = KMeans(5, random_state = 123)
    kmeans5.fit(X_train)
    train["n_location_cluster_k5"] = kmeans5.predict(X_train)
    train["s_location_cluster_k5"] = 'cluster_' + (train.n_location_cluster_k5).astype(str)
    test["n_location_cluster_k5"] = kmeans5.predict(X_test)
    test["s_location_cluster_k5"] = 'cluster_' + (test.n_location_cluster_k5).astype(str)


    #K of 6
    X_train = train[["latitude", "longitude"]]
    X_test = test[["latitude", "longitude"]]
    kmeans6 = KMeans(6, random_state = 123)
    kmeans6.fit(X_train)
    train["n_location_cluster_k6"] = kmeans6.predict(X_train)
    train["s_location_cluster_k6"] = 'cluster_' + (train.n_location_cluster_k6).astype(str)
    test["n_location_cluster_k6"] = kmeans6.predict(X_test)
    test["s_location_cluster_k6"] = 'cluster_' + (test.n_location_cluster_k6).astype(str)
    
    return train, test, kmeans5, kmeans6


def cluster_year_sqft_roomcount(train_scaled, test_scaled, train, test):
    """Takes in train_scaled and test_scaled dataframes and using Kmeans with 6 of
    return kmeans, train, test"""
    X_train = train_scaled[["calculatedfinishedsquarefeet", "room_count", "yearbuilt"]]
    X_test = test_scaled[["calculatedfinishedsquarefeet", "room_count", "yearbuilt"]]
    kmeans = KMeans(6, random_state = 123)
    kmeans.fit(X_train)
    train["n_size_and_year_cluster"] = kmeans.predict(X_train)
    train["s_size_and_year_cluster"] = 'cluster_' + (train.n_size_and_year_cluster).astype(str)
    test["n_size_and_year_cluster"] = kmeans.predict(X_test)
    test["s_size_and_year_cluster"] = 'cluster_' + (test.n_size_and_year_cluster).astype(str)
    return kmeans, train, test
        

    
#Functions for stats testing    
    
    
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
        print("We fail to reject the null hypothesis for Cluster 0")

    
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


def t_test_year_sqft_roomcount(train):
    """Runs a T-Test for each cluster group overall logerror mean
    alpha is set to a 95% confidence
    returns a p value for each cluster"""
    alpha = .05
    
    loc_0 = train[train.n_size_and_year_cluster == 0]
    t0, p0 = stats.ttest_1samp(loc_0.logerror, train.logerror.mean())
    if p0 < alpha:
        print("We reject the null hypothesis for Cluster 0")
    else:
        print("We fail to reject the null hypothesis for Cluster 0")
    
    loc_1 = train[train.n_size_and_year_cluster == 1]
    t1, p1 = stats.ttest_1samp(loc_1.logerror, train.logerror.mean())
    if p1 < alpha:
        print("We reject the null hypothesis for Cluster 1")
    else:
        print("We fail to reject the null hypothesis for Cluster 1")
        
    loc_2 = train[train.n_size_and_year_cluster == 2]
    t2, p2 = stats.ttest_1samp(loc_2.logerror, train.logerror.mean())
    if p2 < alpha:
        print("We reject the null hypothesis for Cluster 2")
    else:
        print("We fail to reject the null hypothesis for Cluster 2")   
        
    loc_3 = train[train.n_size_and_year_cluster == 3]
    t3, p3 = stats.ttest_1samp(loc_3.logerror, train.logerror.mean())
    if p3 < alpha:
        print("We reject the null hypothesis for Cluster 3")
    else:
        print("We fail to reject the null hypothesis for Cluster 3")
        
    loc_4 = train[train.n_size_and_year_cluster == 4]
    t4, p4 = stats.ttest_1samp(loc_4.logerror, train.logerror.mean())
    if p3 < alpha:
        print("We reject the null hypothesis for Cluster 4")
    else:
        print("We fail to reject the null hypothesis for Cluster 4")
        
    loc_5 = train[train.n_size_and_year_cluster == 5]
    t5, p5 = stats.ttest_1samp(loc_5.logerror, train.logerror.mean())
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



## Functions for visuals


def boxplots_for_logerror_clusters(train):
    """Plots a few of the features against logerror"""
    plt.figure(figsize=(13,10))
    plt.subplot(2,2,1)
    plt.title("Finished SqFt")
    sns.boxplot(data=train, x="calculatedfinishedsquarefeet", y="s_cluster_target")
    plt.subplot(2,2,2)
    plt.title("Lot Size")
    sns.boxplot(data=train, x="lotsizesquarefeet", y="s_cluster_target")
    plt.subplot(2,2,3)
    plt.title("Year Built")
    sns.boxplot(data=train, x="yearbuilt", y="s_cluster_target")
    plt.subplot(2,2,4)
    plt.title("Assessed Home Value")
    sns.boxplot(data=train, x="taxvaluedollarcnt", y="s_cluster_target")
    plt.suptitle("Logerror Clusters against Various Features", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    
def logerror_distro_visualization(df):
    """
    This visualization shows you 4 distrubution plots of the logerrors
    1st = all counties
    2nd = LA county
    3rd = Orange County
    4th = Ventura County
    """
    c="#006AFF"
    zillow = df
    la_county = df[df.County == "Los Angeles"]
    orange_county = df[df.County == "Orange"]
    ventura_county = df[df.County == "Ventura"]
    
    plt.figure(figsize=(12,12))
    sns.distplot(zillow.logerror, color=c)
    plt.xlim(-1, 1)
    plt.ylabel("Count")
    plt.title("Logerror Distribution for all three Counties")

    plt.figure(figsize=(12,12))

    plt.suptitle('Logerror Distribution by County - note Y axis is not on the same scale', fontsize=14)

    plt.subplot(3, 3, 1)
    sns.distplot(la_county.logerror, color=c)
    plt.xlim(-.25, .25)
    plt.ylabel("Count")
    plt.title("Los Angeles")

    plt.subplot(3, 3, 2)
    sns.distplot(orange_county.logerror, color=c)
    plt.xlim(-.25, .25)
    plt.ylabel("Count")
    plt.title("Orange County")

    plt.subplot(3, 3, 3)
    sns.distplot(ventura_county.logerror, color=c)
    plt.xlim(-.25, .25)
    plt.ylabel("Count")
    plt.title("Ventura County")
    
def num_home_by_county(train):
    """
    Returns a graph with the number of homes in each county
    """

    plt.rcParams["figure.figsize"] = (13,10)

    ax = sns.countplot(data=train, x='County', palette='Blues_r', order=['Los Angeles', 'Orange', 'Ventura'])

    for p in ax.patches:
            ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+50))

    plt.title('Number of Homes by County (train dataset)')
    plt.xlabel('County Name')
    plt.ylabel('Number of Homes')
    plt.show()
    
    
def homevalue_distro_visualization(df):
    """
    This visualization shows you 4 distrubution plots of the home values
    1st = all counties
    2nd = LA county
    3rd = Orange County
    4th = Ventura County
    """
    c="#006AFF"
    zillow = df
    la_county = df[df.County == "Los Angeles"]
    orange_county = df[df.County == "Orange"]
    ventura_county = df[df.County == "Ventura"]
    
    plt.figure(figsize=(12,12))
    sns.distplot(zillow.taxvaluedollarcnt, color=c)
    plt.ylabel("Count")
    plt.title("Home Value Distribution for all three Counties")

    plt.figure(figsize=(12,12))

    plt.suptitle('Home Value by County - note Y axis is not on the same scale', fontsize=14)

    plt.subplot(3, 3, 1)
    sns.distplot(la_county.taxvaluedollarcnt, color=c)
    plt.ylabel("Count")
    plt.title("Los Angeles")

    plt.subplot(3, 3, 2)
    sns.distplot(orange_county.taxvaluedollarcnt, color=c)
    plt.ylabel("Count")
    plt.title("Orange County")

    plt.subplot(3, 3, 3)
    sns.distplot(ventura_county.taxvaluedollarcnt, color=c)
    plt.ylabel("Count")
    plt.title("Ventura County")
    
    
# Summaries

def home_value_summary(train):
    """
    Shows us a summary of mean, median, max, min, 
    and std for the 3 counties
    """
    
    #get mean for each county
    county_mean = pd.DataFrame(train.groupby("County").taxvaluedollarcnt.mean())
    county_mean.columns = ['Mean Home Value']
    county_mean['Mean Home Value'] = county_mean['Mean Home Value'].round().astype(int) 
    
    #get median for each county
    county_median = pd.DataFrame(train.groupby("County").taxvaluedollarcnt.median())
    county_median.columns = ['Median Home Value']
    county_median['Median Home Value'] = county_median['Median Home Value'].round().astype(int)
    
    #get max for each county    
    county_max = pd.DataFrame(train.groupby("County").taxvaluedollarcnt.max())
    county_max.columns = ['Max Home Value']
    county_max['Max Home Value'] = county_max['Max Home Value'].round().astype(int)
    
    #get min for each county       
    county_min = pd.DataFrame(train.groupby("County").taxvaluedollarcnt.min())
    county_min.columns = ['Min Home Value']
    county_min['Min Home Value'] = county_min['Min Home Value'].round().astype(int)
    
    
    #get STD for each county     
    county_std = pd.DataFrame(train.groupby("County").taxvaluedollarcnt.std())
    county_std.columns = ['STD of Home Values']
    county_std['STD of Home Values'] = county_std['STD of Home Values'].round().astype(int)
    
    #merge all info into one summary
    summary1 = pd.merge(county_mean, county_median, left_index=True, right_index=True)
    summary2 = pd.merge(county_max , county_min , left_index=True, right_index=True)
    summary3 = pd.merge(summary1 , summary2 , left_index=True, right_index=True)
    summary = pd.merge(summary3 , county_std , left_index=True, right_index=True)
    
    return summary


def logerror_summary(train):
    """
    Shows us a summary of mean, median, max, min, 
    and std for the 3 counties
    """
    
    #get mean for each county
    county_mean = pd.DataFrame(train.groupby("County").logerror.mean())
    county_mean.columns = ['Mean Logerror']

    
    #get median for each county
    county_median = pd.DataFrame(train.groupby("County").logerror.median())
    county_median.columns = ['Median Logerror']

    
    #get max for each county    
    county_max = pd.DataFrame(train.groupby("County").logerror.max())
    county_max.columns = ['Max Logerror']

    
    #get min for each county       
    county_min = pd.DataFrame(train.groupby("County").logerror.min())
    county_min.columns = ['Min Logerror']

    
    
    #get STD for each county     
    county_std = pd.DataFrame(train.groupby("County").logerror.std())
    county_std.columns = ['STD of Logerror']

    
    #merge all info into one summary
    summary1 = pd.merge(county_mean, county_median, left_index=True, right_index=True)
    summary2 = pd.merge(county_max , county_min , left_index=True, right_index=True)
    summary3 = pd.merge(summary1 , summary2 , left_index=True, right_index=True)
    summary = pd.merge(summary3 , county_std , left_index=True, right_index=True)
    
    return summary