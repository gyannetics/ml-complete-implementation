import sys
from pandas import DataFrame
from src.exceptions import CustomException
from src.utils import load_object
import os
from src.logger import logging


class PredictPipeline:
    """
    Class responsible for loading a trained model and a preprocessor, and using them to make predictions.

    Methods:
        predict(features): Takes input features and returns the model's predictions.
    """

    def __init__(self) -> None:
        """
        Initializes the PredictPipeline instance.
        """
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features: DataFrame):
        """
        Predicts the target variable based on input features using the pre-trained model.

        Args:
            features (pd.DataFrame): The input features for prediction.

        Returns:
            The prediction results from the model.

        Raises:
            CustomException: For handling exceptions during the prediction process.
        """
        try:
            logging.info("@Predict_Pipeline: Loading model and preprocessor")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            logging.info(
                "@Predict_Pipeline: Model and preprocessor loaded successfully")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A class to encapsulate input data for prediction.

    Attributes:
        gender (str): Gender of the individual.
        race_ethnicity (str): Race/Ethnicity group.
        parental_level_of_education (str): Parental level of education.
        lunch (str): Type of lunch.
        test_preparation_course (str): Test preparation course status.
        reading_score (int): Reading score.
        writing_score (int): Writing score.

    Methods:
        get_data_as_data_frame(): Converts the instance data into a pandas DataFrame.
    """

    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> DataFrame:
        """
        Converts the instance data into a pandas DataFrame for prediction.

        Returns:
            pd.DataFrame: DataFrame containing the data from the instance.

        Raises:
            CustomException: For handling exceptions during DataFrame creation.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
