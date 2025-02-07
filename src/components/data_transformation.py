import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig: 
  preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()
    self.numerical_columns = ["writing_score", "reading_score"]
    self.categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
    self.target_column = "math_score"
  def get_data_transformer_object(self):
    '''
      This function is responsible for data transformation.
    '''

    try:
      num_pipeline=Pipeline(
        steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler())
        ]
      )
      cat_pipeline=Pipeline(
        steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("encoder", OneHotEncoder()),
        ]
      )

      logging.info("Numerical columns scaling complete!")
      logging.info("Categorical columns encoding complete!")

      preprocessor = ColumnTransformer(
        [
          ("num_pipeline", num_pipeline, self.numerical_columns),
          ("cat_pipeline", cat_pipeline, self.categorical_columns)
        ]
      )
      
      return preprocessor

    except Exception as e:
      raise CustomException(e, sys)
  
  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info("Read train and test datasets as dataframe complete")

      logging.info("Obtaining preprocessing object; Initiating data transformation")

      preprocessing_obj = self.get_data_transformer_object()

      input_feature_train_df = train_df.drop(columns=[self.target_column],axis=1)
      target_feature_train_df = train_df[self.target_column]

      input_feature_test_df = test_df.drop(columns=[self.target_column],axis=1)
      target_feature_test_df = test_df[self.target_column]

      logging.info("Applying preprocessing object to train and test datasets")

      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
      
      train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
      test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

      logging.info("Saved preprocessor object to file")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
      )

      return (
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
      )
    except Exception as e:
      raise CustomException(e, sys)
