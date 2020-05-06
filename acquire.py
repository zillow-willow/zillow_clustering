import pandas as pd
from env import user, password, host
from os import path

def get_url(database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

def get_from_sql():
    query = '''
    SELECT prop.*, 
           pred1.logerror, 
           pred1.transactiondate, 
           air.airconditioningdesc, 
           arch.architecturalstyledesc, 
           build.buildingclassdesc, 
           heat.heatingorsystemdesc, 
           landuse.propertylandusedesc, 
           story.storydesc, 
           construct.typeconstructiondesc 
    FROM   properties_2017 prop 
           LEFT JOIN predictions_2017 pred1 USING (parcelid) 
           INNER JOIN (SELECT parcelid, 
                              Max(transactiondate) maxtransactiondate 
                       FROM   predictions_2017 
                       GROUP  BY parcelid) pred2 
                   ON pred1.parcelid = pred2.parcelid 
                      AND pred1.transactiondate = pred2.maxtransactiondate 
           LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
           LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
           LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
           LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
           LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
           LEFT JOIN storytype story USING (storytypeid) 
           LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
    WHERE  prop.latitude IS NOT NULL 
           AND prop.longitude IS NOT NULL;
    '''

    url = get_url('zillow')

    zillow = pd.read_sql(query, url)

    zillow.to_csv('zillow_acquire.csv')
    
def drop_columns(df):
    columns_to_drop = ['architecturalstyletypeid', 'airconditioningtypeid','basementsqft', 
                       'buildingclasstypeid', 'decktypeid', 'finishedfloor1squarefeet',
                       'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 
                       'finishedsquarefeet6', 'fireplacecnt', 'hashottuborspa', 
                       'heatingorsystemtypeid','poolsizesum','pooltypeid10', 'pooltypeid2', 
                       'pooltypeid7', 'regionidneighborhood', 'storytypeid', 
                       'threequarterbathnbr', 'typeconstructiontypeid','yardbuildingsqft17', 
                       'yardbuildingsqft26', 'numberofstories', 'fireplaceflag', 'taxdelinquencyflag', 
                       'taxdelinquencyyear', 'architecturalstyledesc', 'buildingclassdesc', 
                       'storydesc', 'typeconstructiondesc']
    
    df = df.drop(columns=columns_to_drop)
    
    return df
    
def get_data():
    if not path.isfile("zillow_acquire.csv"):
        get_from_sql()
        
    zillow = pd.read_csv('zillow_acquire.csv', index_col='Unnamed: 0')
    
    zillow = drop_columns(zillow)
        
    return zillow