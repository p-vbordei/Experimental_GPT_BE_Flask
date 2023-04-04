import csv
import io
import json
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import traceback

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"], methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"])

api = Api(app)

class ChatGPT(Resource):
    @cross_origin()
    def post(self):
        print(request.method)
        print(request.headers)
        print(request.data)

        # Parse the incoming JSON data and store the text in the 'input_text' variable
        json_data = request.get_json()
        input_text = json_data['text']
        print(f'Input text: {input_text}')

        try:
            csv_file = request.data.decode("utf-8")
            with open('received.csv', 'w', encoding='utf-8') as f:
                f.write(csv_file)

            results = "Sample response" # Define 'results' variable as it was not defined in your code
            return {"results": results}
        except Exception as e:
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            error_message = ''.join(tb_str)
            print(error_message)  # Add this line to print the traceback to the terminal
            return {"error": error_message}, 500
        


api.add_resource(ChatGPT, '/webhook')

if __name__ == '__main__':
    app.run(debug=True, port = 8080)

"""
conda activate oaie
cd Documents/Development/oaie1

python app.py
ngrok http 8080
"""