import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler , StandardScaler , RobustScaler , MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest , chi2

def create_preprocessor():

    trf1 = ColumnTransformer([ 
        ('impute_age', SimpleImputer(), [2]),          # Age
        ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])  # Embarked
    ], remainder='passthrough')
    
    trf2 = ColumnTransformer([
        ('ohe_sex_embarked', OneHotEncoder(sparse_output=False, handle_unknown='ignore') , [1,3])
    ], remainder='passthrough')

    trf3 = ColumnTransformer([
        ('scale' , MinMaxScaler() , slice(0,10))
    ])
    
    trf4 = SelectKBest(score_func=chi2 , k=10)

    preprocessor = Pipeline([
        ('trf1' , trf1) , 
        ('trf2' , trf2) , 
        ('trf3' , trf3) ,
        ('trf4' , trf4) 
    ])


    return preprocessor

if __name__ == '__main__':

    print("Testing preprocessor creation...")
    preprocessor = create_preprocessor()
    
    print(preprocessor)
