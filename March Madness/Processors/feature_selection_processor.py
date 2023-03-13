import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV

def feature_selection(categorical_variables,numeric_variables, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('scaler',StandardScaler())])

    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numeric_variables),
        ('cat', categorical_transformer, categorical_variables)
    ])

    kfolds = 5

    # The range of penalty levels
    min_alpha = 0.1
    max_alpha = 100
    n_candidates = 1000
    C_list = list(1/np.linspace(min_alpha, max_alpha, num=n_candidates))

    # Model Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', LogisticRegressionCV(Cs=C_list, cv=kfolds, penalty='l1',scoring='neg_log_loss',solver='liblinear', max_iter=2000, random_state=1, n_jobs=-1))
                             ])

    pipeline.fit(X_train,y_train)

    return pipeline