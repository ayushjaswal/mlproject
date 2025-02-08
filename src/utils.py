import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            param = params[list(models.keys())[i]]

            model = list(models.values())[i]

            grid = GridSearchCV(model, param, cv=3)
            grid.fit(X_train, y_train)
            
            # model.fit(X_train, y_train)
            # y_train_pred = model.predict(X_train)
            # y_test_pred = model.predict(X_test)
            
            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report, grid.best_params_
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, sys)
