from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
import rest_prediction
import rest_features

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API for AWARE!'

    def post(self):
        try:
            value = request.get_json()
            if value:
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': str(error)}

class GetPredictionOutput(Resource): # uploading the data through a JSON object
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            data = rest_features.main(data)
            predict = rest_prediction.predict_alc_level(data)
            predictOutput = predict
            return {'predict':predictOutput}

        except Exception as error:
            return {'error': str(error)}

class UploadCSV(Resource): # uploading the data through a CSV file
    def post(self):
        try:
            if 'file' not in request.files:
                return {'error': 'No file part'}, 400

            file = request.files['file']

            # You can save the file or process it directly
            # Example: Save the file
            data = pd.read_csv(file)
            data.to_csv('uploaded_file.csv', index=False)

            # Example: Process the file
            # You can now read the CSV file and perform your machine learning tasks
            rest_features.main()
            rest_prediction.combine_features()
            predict = rest_prediction.predict_alc_level()
            predictOutput = predict[-1]
            return {'predict':predictOutput}, 201

        except Exception as error:
            return {'error': str(error)}

api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')
api.add_resource(UploadCSV, '/uploadCSV')
