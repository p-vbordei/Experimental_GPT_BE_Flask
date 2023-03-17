from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class ChatGPT(Resource):
    def post(self):
        data = request.get_json()
        # Process the data and call ChatGPT API here
        response = {"response": "This is a response from ChatGPT"}

        return response

api.add_resource(ChatGPT, '/webhook')

if __name__ == '__main__':
    app.run(debug=True)