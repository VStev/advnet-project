import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def prep(dataset):
    batch_size = 64
    fe_data = dataset.copy()
    fe_data = fe_data.drop(["Label", "Sub_Cat"], axis = 1) # change accordingly, designed to be used with IOTID-20 dataset
    le = preprocessing.LabelEncoder()
    for column_name in fe_data.columns:
        if fe_data[column_name].dtype == object:
            fe_data[column_name] = le.fit_transform(fe_data[column_name])
        else:
            pass
    fe_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    fe_data.dropna(inplace=True)
    fe_datay = fe_data.Cat
    fe_data = fe_data.drop(["Cat"], axis = 1)
    scaler = StandardScaler()
    mapper = DataFrameMapper([(fe_data.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(fe_data.copy(), 4)
    scaled_features_df = pd.DataFrame(scaled_features, index=fe_data.index, columns=fe_data.columns)
    fe_datax = scaled_features_df
    fex_train, fex_valid, fey_train, fey_valid = train_test_split(fe_datax, fe_datay, test_size = 0.3, random_state = 7)
    return:{
        "x_train" : fex_train,
        "x_test" : fex_valid,
        "y_train" : fey_train,
        "y_test" : fey_valid
    }