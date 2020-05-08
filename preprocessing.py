import pandas as pd
import numpy as np

#exploration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_month(row):
    return row[5:7]

def get_day(row):
    return row[-2:]

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
    df['trans_month'] = df.transactiondate.apply(lambda row: get_month(row))
    df['trans_day'] = df.transactiondate.apply(lambda row: get_day(row))

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


def scale_columns(train_df, test_df):
    """ 
    Scales all numeric columns in train and test using a MinMax scaler
    returns the scaler, scaled dataframes for train and test.
    """
    train_scaled_df = train_df.copy()
    test_scaled_df = test_df.copy()
    scaler = MinMaxScaler()
    numeric_columns = list(train_scaled_df.select_dtypes('number').columns)
    train_scaled_df[numeric_columns] = scaler.fit_transform(train_scaled_df[numeric_columns])
    test_scaled_df[numeric_columns] = scaler.transform(test_scaled_df[numeric_columns])
    return scaler, train_scaled_df, test_scaled_df

