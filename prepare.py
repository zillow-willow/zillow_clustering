import pandas as pd
from acquire import get_data

def drop_columns(df):
    columns_to_drop = ['id', 'parcelid', 'propertylandusetypeid', 'propertylandusedesc', 
                      'calculatedbathnbr', 'buildingqualitytypeid', 'fullbathcnt', 'assessmentyear', 
                       'propertyzoningdesc', 'propertycountylandusecode', 'rawcensustractandblock', 
                       'airconditioningdesc', 'heatingorsystemdesc', 'regionidcity', 'regionidzip', 
                       'roomcnt', 'unitcnt', 'censustractandblock']
    df = df.drop(columns=columns_to_drop)
    return df

def impute_median(df):
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median())
    df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(df.finishedsquarefeet12.median())
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median())
    df.yearbuilt = df.yearbuilt.fillna(df.yearbuilt.median())
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(df.structuretaxvaluedollarcnt.median)
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(df.taxvaluedollarcnt.median())
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(df.landtaxvaluedollarcnt.median())
    df.taxamount = df.taxamount.fillna(df.taxamount.median())
    
    return df

def pool_and_garage(df):
    if (df.garagecarcnt > 0) and (df.poolcnt > 0):
        return 'Garage and Pool'
    elif df.garagecarcnt > 0:
        return 'Garage Only'
    elif df.poolcnt > 0:
        return 'Pool Only'
    else:
        return 'No Pool or Garage'
    
def combine_garage_and_pool(df):
    df['extras'] = df.apply(pool_and_garage, axis=1)
    df = df.drop(columns=['garagecarcnt', 'poolcnt'])
    return df

def create_new_features(df):
    df = combine_garage_and_pool(df)
    return df
    
def wrangle_data():
    zillow = get_data()
    zillow = drop_columns(zillow)
    zillow = impute_median(zillow)
    zillow = create_new_features(zillow)
    zillow = zillow[(zillow.bathroomcnt > 0) & (zillow.bedroomcnt > 0)]
    
    return zillow