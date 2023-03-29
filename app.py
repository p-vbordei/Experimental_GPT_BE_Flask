import csv
import io
import openai
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import dataPrep
import gptQuerry

# Replace with your OpenAI API key
openai.api_key = "your-api-key"

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"], methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"])

api = Api(app)

class ChatGPT(Resource):
    @cross_origin()
    def post(self):
        print(request.method)
        print(request.headers)
        print(request.data)

        raise Exception("Debugging")  # Add this line

        csv_file = request.data.decode("utf-8")
        with open('received.csv', 'w', encoding='utf-8') as f:
            f.write(csv_file)

        # Define 'results' variable as it was not defined in your code
        results = "Sample response"

        return {"results": results}

api.add_resource(ChatGPT, '/webhook')

if __name__ == '__main__':
    app.run(debug=True)

"""
conda activate oaie
cd Documents/Development/oaie1

python app.py
ngrok http 5000
"""