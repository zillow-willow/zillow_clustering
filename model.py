import pandas as pd

import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def create_model_without_clusters(train, val):
    '''
    Takes in the train and validation set, fits a model, prints out the metrics of the model, and returns nothing.
    '''
    Xtrain = train[['room_count', 'yearbuilt', 'tax_rate', 'dollar_per_sqft_home', 
                    'dollar_per_sqft_land', 'lotsizesquarefeet']]
    ytrain = train[['logerror']]

    Xval = val[['room_count', 'yearbuilt', 'tax_rate', 'dollar_per_sqft_home', 
                'dollar_per_sqft_land', 'lotsizesquarefeet']]
    yval = val[['logerror']]

    predictions = pd.DataFrame(yval)
    predictions['baseline_mean'] = ytrain.logerror.mean()

    lm = LinearRegression()
    lm.fit(Xtrain, ytrain)

    predictions[f'lr_best_model_no_clusters'] = lm.predict(Xval)

    print('RMSE:')
    print(predictions.drop(columns='logerror').apply(lambda col: math.sqrt(mean_squared_error(predictions.logerror, col))).sort_values())
    
    baseline_RMSE = math.sqrt(mean_squared_error(predictions.logerror, predictions[f'baseline_mean']))
    model_RMSE = math.sqrt(mean_squared_error(predictions.logerror, predictions['lr_best_model_no_clusters']))
    model_diff = baseline_RMSE - model_RMSE
    
    print('\n')
    print(f'The model beat the baseline by {model_diff:.5f}')
    
    
    

def create_model_with_logerror_clusters(train, val):
    '''
    Takes in the train and validation set, fits a model, prints out the metrics of the model, and returns nothing.
    '''
    prediction_diffs = []

    #Loop through each cluster, create a model, and print metrics
    for i in range(0, 6):
        # Prepare train and vals for modeling
        Xtrain = train[train.n_cluster_target == i][['room_count', 'yearbuilt', 'tax_rate', 
                                                     'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                                     'lotsizesquarefeet']]

        ytrain = train[train.n_cluster_target == i][['logerror']]

        Xval = val[val.n_cluster_target == i][['room_count', 'yearbuilt', 'tax_rate', 
                                               'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                               'lotsizesquarefeet']]

        yval = val[val.n_cluster_target == i][['logerror']]

        # Create the model
        lm = LinearRegression()
        lm.fit(Xtrain, ytrain)

        #Create a DataFrame with actuals, baselines, and predictions
        cluster_predictions = pd.DataFrame(yval.logerror)
        cluster_predictions[f'cluster_{i}_baseline_mean'] = yval.logerror.mean()
        cluster_predictions[f'cluster_{i}_lr_predictions'] = lm.predict(Xval)

        baseline_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_baseline_mean']))
        model_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_lr_predictions']))
        prediction_diffs.append(model_RMSE - baseline_RMSE)

        print(f'Logerror Cluster {i} RMSE:')

        # Print out the RMSE for each column in the predictions
        print(cluster_predictions.drop(columns='logerror')
                                 .apply(lambda col: 
                                            math.sqrt(
                                                mean_squared_error(
                                                    cluster_predictions.logerror, col)
                                            )
                                       )
                                 .sort_values()
             )

        if baseline_RMSE > model_RMSE:
            print(f'The model beat the baseline by {baseline_RMSE - model_RMSE:.5f}')
        else:
            print(f'The baseline beat the model by {model_RMSE - baseline_RMSE:.5f}')

        print('\n')

    print(f'The models, on average, were beat by the baseline by {sum(prediction_diffs)/len(prediction_diffs):.5f}')
    
    
    
    
def create_model_with_location_clusters(train, val):
    '''
    Takes in the train and validation set, fits a model, prints out the metrics of the model, and returns nothing.
    '''
    prediction_diffs = []

    #Loop through each cluster, create a model, and print metrics
    for i in range(0, 6):
        # Prepare train and vals for modeling
        Xtrain = train[train.n_location_cluster_k6 == i][['room_count', 'yearbuilt', 'tax_rate', 
                                                     'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                                     'lotsizesquarefeet']]

        ytrain = train[train.n_location_cluster_k6 == i][['logerror']]

        Xval = val[val.n_location_cluster_k6 == i][['room_count', 'yearbuilt', 'tax_rate', 
                                               'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                               'lotsizesquarefeet']]

        yval = val[val.n_location_cluster_k6 == i][['logerror']]

        # Create the model
        lm = LinearRegression()
        lm.fit(Xtrain, ytrain)

        #Create a DataFrame with actuals, baselines, and predictions
        cluster_predictions = pd.DataFrame(yval.logerror)
        cluster_predictions[f'cluster_{i}_baseline_mean'] = yval.logerror.mean()
        cluster_predictions[f'cluster_{i}_lr_predictions'] = lm.predict(Xval)

        baseline_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_baseline_mean']))
        model_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_lr_predictions']))
        prediction_diffs.append(baseline_RMSE - model_RMSE)

        print(f'Location Cluster {i} RMSE:')

        # Print out the RMSE for each column in the predictions
        print(cluster_predictions.drop(columns='logerror')
                                 .apply(lambda col: 
                                            math.sqrt(
                                                mean_squared_error(
                                                    cluster_predictions.logerror, col)
                                            )
                                       )
                                 .sort_values()
             )

        if baseline_RMSE > model_RMSE:
            print(f'The model beat the baseline by {baseline_RMSE - model_RMSE:.5f}')
        else:
            print(f'The baseline beat the model by {model_RMSE - baseline_RMSE:.5f}')

        print('\n')

    print(f'The models, on average, beat the baseline by {sum(prediction_diffs)/len(prediction_diffs):.5f}')
    
    
    
    
def create_model_with_size_and_year_clusters(train, val):
    '''
    Takes in the train and validation set, fits a model, prints out the metrics of the model, and returns nothing.
    '''
    prediction_diffs = []

    #Loop through each cluster, create a model, and print metrics
    for i in range(0, 6):
        # Prepare train and vals for modeling
        Xtrain = train[train.n_size_and_year_cluster == i][['room_count', 'yearbuilt', 'tax_rate', 
                                                     'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                                     'lotsizesquarefeet']]

        ytrain = train[train.n_size_and_year_cluster == i][['logerror']]

        Xval = val[val.n_size_and_year_cluster == i][['room_count', 'yearbuilt', 'tax_rate', 
                                               'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                               'lotsizesquarefeet']]

        yval = val[val.n_size_and_year_cluster == i][['logerror']]

        # Create the model
        lm = LinearRegression()
        lm.fit(Xtrain, ytrain)

        #Create a DataFrame with actuals, baselines, and predictions
        cluster_predictions = pd.DataFrame(yval.logerror)
        cluster_predictions[f'cluster_{i}_baseline_mean'] = yval.logerror.mean()
        cluster_predictions[f'cluster_{i}_lr_predictions'] = lm.predict(Xval)

        baseline_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_baseline_mean']))
        model_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_lr_predictions']))
        prediction_diffs.append(model_RMSE - baseline_RMSE)

        print(f'Size and Year Cluster {i} RMSE:')

        # Print out the RMSE for each column in the predictions
        print(cluster_predictions.drop(columns='logerror')
                                 .apply(lambda col: 
                                            math.sqrt(
                                                mean_squared_error(
                                                    cluster_predictions.logerror, col)
                                            )
                                       )
                                 .sort_values()
             )

        if baseline_RMSE > model_RMSE:
            print(f'The model beat the baseline by {baseline_RMSE - model_RMSE:.5f}')
        else:
            print(f'The baseline beat the model by {model_RMSE - baseline_RMSE:.5f}')

        print('\n')

    print(f'The models, on average, were beat by the baseline by {sum(prediction_diffs)/len(prediction_diffs):.5f}')
    
    
    
    
def create_test_model_with_location_cluster(train, test):
    '''
    Takes in the train and test set, fits a model, prints out the metrics of the model, and returns nothing.
    '''
    prediction_diffs = []

    #Loop through each cluster, create a model, and print metrics
    for i in range(0, 6):
        # Prepare train and vals for modeling
        Xtrain = train[train.n_location_cluster_k6 == i][['room_count', 'yearbuilt', 'tax_rate', 
                                                     'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                                     'lotsizesquarefeet']]

        ytrain = train[train.n_location_cluster_k6 == i][['logerror']]

        Xtest = test[test.n_location_cluster_k6 == i][['room_count', 'yearbuilt', 'tax_rate', 
                                               'dollar_per_sqft_home', 'dollar_per_sqft_land', 
                                               'lotsizesquarefeet']]

        ytest = test[test.n_location_cluster_k6 == i][['logerror']]

        # Create the model
        lm = LinearRegression()
        lm.fit(Xtrain, ytrain)

        #Create a DataFrame with actuals, baselines, and predictions
        cluster_predictions = pd.DataFrame(ytest.logerror)
        cluster_predictions[f'cluster_{i}_baseline_mean'] = ytest.logerror.mean()
        cluster_predictions[f'cluster_{i}_lr_predictions'] = lm.predict(Xtest)

        baseline_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_baseline_mean']))
        model_RMSE = math.sqrt(mean_squared_error(cluster_predictions.logerror, cluster_predictions[f'cluster_{i}_lr_predictions']))
        prediction_diffs.append(baseline_RMSE - model_RMSE)

        print(f'Location Cluster {i} RMSE:')

        # Print out the RMSE for each column in the predictions
        print(cluster_predictions.drop(columns='logerror')
                                 .apply(lambda col: 
                                            math.sqrt(
                                                mean_squared_error(
                                                    cluster_predictions.logerror, col)
                                            )
                                       )
                                 .sort_values()
             )

        if baseline_RMSE > model_RMSE:
            print(f'The model beat the baseline by {baseline_RMSE - model_RMSE:.5f}')
        else:
            print(f'The baseline beat the model by {model_RMSE - baseline_RMSE:.5f}')

        print('\n')

    print(f'The models, on average, beat the baseline by {sum(prediction_diffs)/len(prediction_diffs):.5f}')