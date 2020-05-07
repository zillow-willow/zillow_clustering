import pandas as pd
from acquire import get_data

def drop_columns(df):
    '''Takes in the zillow DataFrame and removes specific columns'''
    columns_to_drop = ['id', 'parcelid', 'propertylandusetypeid', 'propertylandusedesc', 
                      'calculatedbathnbr', 'buildingqualitytypeid', 'fullbathcnt', 'assessmentyear', 
                       'propertyzoningdesc', 'propertycountylandusecode', 'rawcensustractandblock', 
                       'airconditioningdesc', 'heatingorsystemdesc', 'regionidcity', 'regionidzip', 
                       'roomcnt', 'unitcnt', 'censustractandblock', 'regionidcounty']
    df = df.drop(columns=columns_to_drop)
    return df

def impute_median(df):
    '''Takes in the zillow DataFrame and imputes the median onto specific columns'''
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median())
    df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(df.finishedsquarefeet12.median())
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median())
    df.yearbuilt = df.yearbuilt.fillna(df.yearbuilt.median())
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df.structuretaxvaluedollarcnt.median())
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(df.taxvaluedollarcnt.median())
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(df.landtaxvaluedollarcnt.median())
    df.taxamount = df.taxamount.fillna(df.taxamount.median())
    
    return df

def pool_and_garage(df):
    '''Takes in the zillow DataFrame and checks the pool and garage status of the property and returns a string catagory'''
    if (df.garagecarcnt > 0) and (df.poolcnt > 0):
        return 'Garage and Pool'
    elif df.garagecarcnt > 0:
        return 'Garage Only'
    elif df.poolcnt > 0:
        return 'Pool Only'
    else:
        return 'No Pool or Garage'
    
def combine_garage_and_pool(df):
    '''Takes in the zillow DataFrame and adds a column that denotes whether a property has a garage or pool. The garage and pool columns are then removed to reduce redundency and the DataFrame is returned.'''
    df['extras'] = df.apply(pool_and_garage, axis=1)
    df = df.drop(columns=['garagecarcnt', 'poolcnt'])
    return df

def label_county(row):
    '''Takes in a single row of the zillow DataFrame and returns the county name based on the fips column'''
    if row['fips'] == 6037:
        return 'Los Angeles'
    elif row['fips'] == 6059:
        return 'Orange'
    elif row['fips'] == 6111:
        return 'Ventura'

def create_county(df):
    '''Creates a county column on zillow DataFrame and removes fips column'''
    df['County'] = df.apply(lambda row: label_county(row), axis=1)
    df = df.drop(columns='fips')
    return df

def create_new_features(df):
    '''Adds new columns onto the zillow DataFrame'''
    df = combine_garage_and_pool(df)
    df = create_county(df)
    return df

def handle_outliers(df):
    '''Takes in the zillow DataFrame, removes row that were determined to be outliers and returns the DataFrame'''
    df = df[df.bathroomcnt <= 6]
    df = df[df.bedroomcnt <= 7]
    df = df[df.calculatedfinishedsquarefeet <= 5382]
    df = df[df.finishedsquarefeet12 <= 5400]
    df = df[df.lotsizesquarefeet <= 17767]
    df = df[df.structuretaxvaluedollarcnt <= 671826]
    df = df[df.taxvaluedollarcnt <= 1892955]
    df = df[df.landtaxvaluedollarcnt <= 1404318]

    return df

def wrangle_data():
    '''Takes no arguments and returns a prepared zillow DataFrame'''
    zillow = get_data()
    zillow = drop_columns(zillow)
    zillow = impute_median(zillow)
    zillow = create_new_features(zillow)
    zillow = zillow[(zillow.bathroomcnt > 0) & (zillow.bedroomcnt > 0)]
    zillow = handle_outliers(zillow)
    
    return zillow