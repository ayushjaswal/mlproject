# System Modules
import os
import sys 
from dataclasses import dataclass

# Working Modules
import numpy as np
import pandas as pd
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Custom Modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join('artifact',"model.pkl")

class ModelTrainer:
  def __init__(self): 
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_arr, test_arr):
    try:
      logging.info("Splitting train and test data")
      X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

      models ={
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Cat Boost Regressor": CatBoostRegressor(verbose=False),
        "XGBoost Regressor": XGBRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "K Nearest Neighbors Regressor": KNeighborsRegressor()
      }

      model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
      logging.info("Training completed!") 

      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

      best_model = models[best_model_name]

      if best_model_score < 0.6: 
        raise CustomException("The model's performance is not satisfactory. Please consider using a different model or preprocessing techniques.") 

      logging.info(f"Best Model: {best_model_name}, Score: {best_model_score}")

      save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model) 
      logging.info("Model saved successfully!")

      predicted_ = best_model.predict(X_test)
      r2_squared = r2_score(y_test, predicted_)

      return r2_squared

    except Exception as e:
      raise CustomException(e, sys)