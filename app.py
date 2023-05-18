import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Load the trained model
clf = joblib.load('trained_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    # Preprocess the input data if needed
    df = preprocess_data(data)

    # Make predictions using the loaded model
    probabilities = clf.predict_proba(df)

    # Get the class labels and corresponding probabilities for the top n predictions
    n = 5  # Specify the number of top predictions to return
    top_n_indices = np.argsort(-probabilities, axis=1)[:, :n]
    top_n_labels = clf.classes_[top_n_indices]
    top_n_probabilities = np.take_along_axis(probabilities, top_n_indices, axis=1)

    # Return the top n predictions as the API response
    response = {'predictions': []}
    for i in range(len(df)):
        predictions = []
        for j in range(n):
            prediction = {
                'label': top_n_labels[i, j],
                'probability': top_n_probabilities[i, j]
            }
            predictions.append(prediction)
        response['predictions'].append(predictions)

    return jsonify(response)


def preprocess_data(data):
    # Convert JSON data to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Split the cpvCode column
    df['cpv'] = df['cpvCode'].str.split('-', expand=True)[0]
    df['cpvCategory'] = df['cpv'].str.slice(stop=5)

    df = df.drop([
        'cpv',
    ], axis=1)

    # Apply Label Encoding to the categorical columns
    columns_to_encode = [
        'cpvCode'
        'noticeNo',
        'title',
        'publicationDate',
        'contractingAuthorityName',
        'cpvCodeName',
        'cNoticeNo',
        'cNoticePublicationDate',
        'cNoticeTitle',
        'cpvCategory',
        'countyCode',
    ]

    # Apply label encoding to the other categorical columns
    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])

    return df


if __name__ == '__main__':
    app.run()
