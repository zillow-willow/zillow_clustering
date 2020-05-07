import pandas as pd
import numpy as np

#exploration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def new_features(df):
    """
    Returns zillow dataframe with the following new features:
    room_count = bathroom and bedroom counts combined
    acres = lot size in acre
    dollar_per_sqft_land = cost of land per sqft
    tax rate = percentage rate for tax
    avg_sqft_per_room = sqft per room (total sqft divided by room count)
    dollar_per_sqft_home = cost of the home per sqft
    """
    df["room_count"] = df.bathroomcnt + df.bedroomcnt
    df["acres"] = df.lotsizesquarefeet / 43560
    df["dollar_per_sqft_land"] = df.landtaxvaluedollarcnt/ df.lotsizesquarefeet
    df["tax_rate"] = df.taxamount/ df.taxvaluedollarcnt
    df["avg_sqft_per_room"] = df.calculatedfinishedsquarefeet / df.room_count
    df["dollar_per_sqft_home"] = df.structuretaxvaluedollarcnt / df.finishedsquarefeet12
    df.drop(columns=["bathroomcnt" , "bedroomcnt"], inplace=True)

    return df



def split_data(df, train_size=.7, seed=123):
    """
    Given a dataframe, train size (optional) and a random state (optional).
    Splits dataframe into test and train dataframes.
    If not provided random state defaults to 123 and
    train_size defaults to .7"""
    # Create the train and test sets
    train, test = train_test_split(df, train_size=train_size, random_state=seed)

    return train, test


def scale_columns(train, test):
    """ 
    Scales all numeric columns in train and test using a MinMax scaler
    returns the scaler, scaled dataframes for train and test.
    """
    scaler = MinMaxScaler()
    numeric_columns = list(train.select_dtypes('number').columns)
    train[numeric_columns] = scaler.fit_transform(train[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])
    return scaler, train, test

