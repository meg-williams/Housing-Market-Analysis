from re import VERBOSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
import dill as pickle 
import tensorflow as tf
from keras.models import Sequential, load_model 
from keras.layers import Dense, Input

# load_data

def load_data(file): 
    df = pd.read_csv(file)
    return df

# split_data

def split_data(x, y): 
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)
    return xtrain, xtest, ytrain, ytest

# clean_data

def clean_data(df): 
    for col in df.columns: 
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']: 
                mean = df[col].mean()
                df[col] = df[col].fillna(mean)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode)[0]
    return df

# encode_data
def encode_data(df, data_type, target_col):

    ohe_cols = []
    target_cols = []
    ord_qual_cols = []

    # training 
    if data_type == 'train':
        for col in df:
            if df[col].dtype == 'object':
                # ordinal encoding 
                if col.endswith(('Qual', 'Cond', 'QC', 'Qu')):
                    ord_qual_cols.append(col)
            
                elif col == 'LandSlope': 
                    ord_LandSlope = ce.OrdinalEncoder(mapping=[{'col': col, 'mapping': {'Gtl': 3, 'Mod': 2, 'Sev': 1}}])
                    ord_LandSlope.fit(df[col])
                    df[col] = ord_LandSlope.transform(df[col])

                elif col == 'Functional': 
                    ord_Functional = ce.OrdinalEncoder(mapping=[{'col': col, 'mapping': {'Typ':8, 'Min1':7, 'Min2': 6, 'Mod':5, 'Maj1': 4, 
                                                                                'Maj2':3, 'Sev': 2, 'Sal': 1 }}])
                    ord_Functional.fit(df[col])
                    df[col] = ord_Functional.transform(df[col])
                # onehote columns 
                elif df[col].nunique() <= 6: 
                    ohe_cols.append(col)
                # target columns 
                else: 
                    target_cols.append(col)

        with open('ord_qual_cols.pickle', 'wb') as f: 
            pickle.dump(ord_qual_cols, f)

        with open('ord_landslope.pickle', 'wb') as f: 
            pickle.dump(ord_LandSlope, f)

        with open('ord_functional.pickle', 'wb') as f: 
            pickle.dump(ord_Functional, f)

        with open('ohe_cols.pickle', 'wb') as f: 
            pickle.dump(ohe_cols, f)
        
        with open('target_cols.pickle', 'wb') as f: 
            pickle.dump(target_cols, f)

        mappings = [{'col': col, 'mapping': {'Ex':5, 'Gd':4, 'TA': 3, 'Fa':2, 'Po': 1}} for col in ord_qual_cols]
        ord_qual = ce.OrdinalEncoder(mapping=mappings)
        ord_qual.fit(df[ord_qual_cols])
        transformed = ord_qual.transform(df[ord_qual_cols])
        df =  pd.concat([df.drop(ord_qual_cols, axis=1), transformed], axis=1)

        with open('ord_qual.pickle', 'wb') as f: 
            pickle.dump(ord_qual, f)

        ohe = ce.OneHotEncoder(handle_unknown='value', use_cat_names=True)
        ohe.fit(df[ohe_cols])
        transformed = ohe.transform(df[ohe_cols])
        df = pd.concat([df.drop(ohe_cols, axis=1), transformed], axis=1)

        with open('hp_ohe.pickle', 'wb') as f: 
            pickle.dump(ohe, f)

        target = ce.TargetEncoder()
        target.fit(df[target_cols], target_col)
        df[target_cols] = target.transform(df[target_cols], target_col)
                
        with open('hp_target.pickle', 'wb') as f: 
            pickle.dump(target, f)

    # test
    elif data_type == 'test': 
        for col in df:
            if df[col].dtype == 'object':
                if col == 'LandSlope': 
                    with open('ord_landslope.pickle', 'rb') as f: 
                        ord_landslope_test = pickle.load(f)
                    df[col] = ord_landslope_test.transform(df[col])

                elif col == 'Functional': 
                    with open('ord_functional.pickle', 'rb') as f: 
                        ord_functional_test = pickle.load(f)
                    df[col] = ord_functional_test.transform(df[col])

        
        with open('ord_qual.pickle', 'rb') as f: 
                        ord_qual_test = pickle.load(f)

        with open('ord_qual_cols.pickle', 'rb') as f: 
             ord_qual_cols_test = pickle.load(f)
                    
        transformed = ord_qual_test.transform(df[ord_qual_cols_test])
        df =  pd.concat([df.drop(ord_qual_cols_test, axis=1), transformed], axis=1)

        with open('ohe_cols.pickle', 'rb') as f: 
            ohe_cols_test = pickle.load(f)
        with open('hp_ohe.pickle', 'rb') as f: 
            ohe_test = pickle.load(f)

        df = pd.concat([df.drop(ohe_cols_test, axis=1), ohe_test.transform(df[ohe_cols_test])], axis=1)

        with open('target_cols.pickle', 'rb') as f: 
            target_cols_test = pickle.load(f)

        with open('hp_target.pickle', 'rb') as f: 
            target_test = pickle.load(f)

        df[target_cols_test] = target_test.transform(df[target_cols_test])
        

    return df




# train_model


def train_model(df, data_type, target_col, target_col_name, id_col, id_col_name, model_file, output_file): 

    if data_type == 'train':
        en = ElasticNet()

        xtrain = df.drop(columns=[target_col, id_col])
        ytrain = df[target_col]


        param_grid = {
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        grid_model = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid, cv=4, n_jobs=-1)

        grid_model.fit(xtrain, ytrain)

        xgb_bestest = grid_model.best_estimator_

        with open(model_file, 'wb') as f: 
            pickle.dump(xgb_bestest, f)

    elif data_type == 'test': 

        with open(model_file, 'rb') as f: 
            model_test = pickle.load(f)

        xtest = df.drop(columns=id_col)
        test_preds = model_test.predict(xtest)

        preds = pd.DataFrame()
        preds.insert(0, id_col_name, df[id_col])
        preds.insert(1, target_col_name, test_preds)

        preds.to_csv(output_file, index=False)



# ann 
def ann(X, y, model, layers, width, activation, output, output_activation, learning_rate, loss, metrics, epochs, batch_size, validation_split): 

    model.add(Input((X.shape[1],)))
    
    for _ in range(layers): 

        model.add(Dense(width, activation=activation))

    model.add(Dense(output, activation=output_activation))

    print(model.summary())

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    return model