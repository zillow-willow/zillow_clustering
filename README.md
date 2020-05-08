# What is affecting the logerror?

To help is figure out what was causing the errors in Zestimates, we have to explore some variables that will help is reduce the logerror.

# How to Reproduce:

- First clone this repo

- ENV.py file with the following information as it pertains to the SQL network (not part of repo):
    - password
    - username
    - host
    
- acquire.py
    - Must include `env.py` file in directory.
    - This file brings in the data from the MySQL Server that the data is stored on
    
- prep.py
    - handles the following:
        - Data Types
        - Missing Values
        - Outliers
        - Erroneous columns/data
        - create new features
        
- preprocessing.py
    - feature engineering
    - splits data into train and test
    - scale numeric data
    
- explore.py
    - Functions for:
        - finding optimal k value for Kmeans
        - elbow plotting
        - clustering features
        - statistical testing
        - visualizations
        
# Deliverables
1. All files necessary to recreate our findings and models
2. Report with analysis, clustering and modeling in .ipynb format
3. GitHub repo containing all files

# Tested Hypothesis
**T-Test for K of 5 on location clustering**
- $H_0$ = There is no difference between the mean logerror scores for cluster 0 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 1 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 2 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 3 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 4 and the overall mean logerror

**T-Test for K of 5 on location clustering**
- $H_0$ = There is no difference between the mean logerror scores for cluster 0 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 1 and the overall mean logerror
- $H_0$ = There is no difference between the mean logerror scores for cluster 2 and the overall mean logerror
- $H_0$  = There is no difference between the mean logerror scores for cluster 3 and the overall mean logerror
- $H_0$  = There is no difference between the mean logerror scores for cluster 4 and the overall mean logerror
- $H_0$  = There is no difference between the mean logerror scores for cluster 6 and the overall mean logerror


## Data Dictionary

| Columns | Definition |
|:--------|-----------:|
|  bathroomcnt | number of bathrooms  |
|  bedroomcnt | number of bedrooms  |
|  calculatedfinishedsquarefeet | SqFt of total living area  |
|  finishedsquarefeet12 | SqFt of finished living area  |
|  latitude | latitude of the middle of the property  |
|  longitude | longitude of the middle of the property  |
|  lotsizesquarefeet | SqFt of the lot  |
|  yearbuilt | Year home was built  |
|  structuretaxvaluedollarcnt | Assessed value of the home structure |
|  taxvaluedollarcnt | Assessed home value |
|  taxamount | tax amount of the home |
|  logerror | logarithmic error of housing price predictions |
|  transactiondate | date sold |
|  extras  | describes if home has a garage, pool, or neither |
|  County  | State County the home is located in |
|  room_count  | Combines bathroomcnt and bedroomcnt into one variable |
|  acres  | Gives lot acreage size |
|  dollar_per_sqft_land  | cost of land per sqft |
|  tax_rate  | percentage rate for taxes |
|  avg_sqft_per_room  | average sqft per room |
|  dollar_per_sqft_home  | cost of structure space per sqft |
|  trans_month  | transaction month |
|  trans_day  | transaction day |

# Project Conclusions
- Location clusters provided the most value
- Clustering size did not perform well in modeling

# Moving Forward
- try to find individual features for each cluster that will help them perform better

# Technical Skills
- Python (including internal and third party libraries)
- SQL
- Hypothesis testing
- Linear Regression, Kmeans

# Data Source for project:
Link may be found [HERE](https://ds.codeup.com/8-clustering/project/)