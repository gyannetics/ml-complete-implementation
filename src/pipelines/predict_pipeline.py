import sys
from pandas import DataFrame
from src.exceptions import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
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
            DataFrame: Pandas DataFrame containing the data from the instance.

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
