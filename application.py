from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

# app = application


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extracting form data
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('ethnicity')
        parental_level_of_education = request.form.get(
            'parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        writing_score_str = request.form.get('writing_score')
        reading_score_str = request.form.get('reading_score')

        # Validating and converting writing_score and reading_score
        if writing_score_str is None or writing_score_str == '':
            return "Writing score is required.", 400
        if reading_score_str is None or reading_score_str == '':
            return "Reading score is required.", 400

        writing_score = float(writing_score_str)
        reading_score = float(reading_score_str)

        # Creating data object
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    application.run(host="0.0.0.0")
