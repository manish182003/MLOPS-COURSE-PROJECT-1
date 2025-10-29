import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data, read_yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing")

            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            categorical_cols = self.config['data_processing']['categorical_features']
            numerical_cols = self.config['data_processing']['numerical_features']

            logger.info("Applying Label Encoding to categorical features")

            le = LabelEncoder()
            mappings = {}

            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])
                mappings[col] = {label: code for label, code in zip(
                    le.classes_, le.transform(le.classes_))}

            logger.info("label Mappings are : ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Doing Skewness Handling")

            skew_threshold = self.config['data_processing']['skewness_threshold']

            skewness = df[numerical_cols].apply(lambda x: x.skew())

            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])

            return df

        except Exception as e:
            logger.error(f"Error during data preprocessing {e}")
            raise CustomException("Data Preprocessing Failed", e)

    def handle_imbalance_data(self, df):
        try:
            logger.info("Handling Imbalanced Data using SMOTE")
            x = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            x_resampled, y_resampled = smote.fit_resample(x, y)

            balanced_df = pd.concat([x_resampled, y_resampled], axis=1)

            logger.info("Imbalanced Data Handling Completed")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during handling imbalance data {e}")
            raise CustomException("Handling Imbalance Data Failed", e)

    def select_features(self, df):
        try:
            logger.info(
                "Starting Feature Selection using RandomForestClassifier")
            x = df.drop(columns=['booking_status'])
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(x, y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame(
                {'Feature': x.columns, 'Importance': feature_importance})

            top_10_features = feature_importance_df.sort_values(
                by='Importance', ascending=False)[:10]

            num_features = self.config['data_processing']['no_of_features']

            top_10_features = feature_importance_df.sort_values(
                by='Importance', ascending=False)[:num_features]
            top_10_df = df[top_10_features.values[:,
                                                  0].tolist() + ["booking_status"]]

            logger.info(
                f"{num_features} features selected: {top_10_features['Feature'].tolist()}")

            logger.info("Feature Selection Completed Successfully")
            return top_10_df

        except Exception as e:
            logger.error(f"Error during feature selection {e}")
            raise CustomException("Feature Selection Failed", e)

    def save_data_to_csv(self, df, path):
        try:
            logger.info("Saving data to processed folder")

            df.to_csv(path, index=False)

            logger.info(f"Data Saved Successfully to {path}")

        except Exception as e:
            logger.info(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)

    def process(self):
        try:
            logger.info("loading data from Raw Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.handle_imbalance_data(train_df)
            test_df = self.handle_imbalance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            self.save_data_to_csv(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data_to_csv(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing completed Successfully")

        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)


if __name__ == "__main__":
    processor = DataProcessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
