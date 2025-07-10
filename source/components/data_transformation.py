import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from source.exception.exception import CustomException
from source.logger.logger import logging
from source.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join("artifacts","proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformaer_object(self):
        try:
            numerical_columns=["reading_score","writing_score"]

            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]


            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical columns : {numerical_columns}")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns : {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)

            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessor=self.get_data_transformaer_object()

            target_column_name="math_score"

            numerical_columns=["reading_score","writing_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessor")


            input_feature_train_array=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor.transform(input_feature_test_df)

            train_array=np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]

            test_array=np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]


            logging.info("Saved preprocessing object")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
                )
        
        except Exception as e:
            raise CustomException(e,sys)
