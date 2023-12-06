import os
import sys
import numpy as np
import pandas as pd
import dill
import time
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exceptions import CustomException
from typing import Dict, Any
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Args:
        file_path (str): The path to the file where the object should be saved.
        obj: The Python object to save.

    Raises:
        CustomException: For handling exceptions that occur during object saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, Any], params: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate multiple models using GridSearchCV and return their performance scores.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Test data features.
        y_test: Test data labels.
        models: Dictionary of model names and their instances.
        params: Dictionary of model names and their corresponding parameter grid for GridSearchCV.

    Returns:
        A dictionary with model names and their respective test R2 scores.

    Raises:
        CustomException: For handling exceptions that occur during the model evaluation.
    """
    try:
        report = {}

        for model_name, model in tqdm(models.items(), desc="Training models"):
            start_time = time.time()  # Start time

            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            end_time = time.time()  # End time
            training_time = end_time - start_time  # Calculate training time

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': gs.best_params_,
                'training_time': training_time,  # Add training time to the report
                'best_estimator': best_model
            }
            # Display current model name
            tqdm.write(
                f"Completed training {model_name}. Time taken: {training_time:.2f} seconds")
            logging.info(
                f"Completed training {model_name}. Time taken: {training_time:.2f} seconds")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using dill.

    Args:
        file_path (str): The path to the file from which to load the object.

    Returns:
        The Python object loaded from the file.

    Raises:
        CustomException: For handling exceptions that occur during object loading.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)