import csv
import io
import openai
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

# Replace with your OpenAI API key
openai.api_key = "your-api-key"

app = Flask(__name__)
CORS(app)  # Enable CORS middleware
api = Api(app)

class ChatGPT(Resource):
    def post(self):
        csv_file = request.data.decode("utf-8")
        csv_reader = csv.reader(io.StringIO(csv_file))

        # Process each row in the CSV
        results = []
        for row in csv_reader:
            review = row[0]  # Assuming the review is in the first column of the CSV
            prompt = f"Analyze the following product review and provide suggestions for improvement: {review}"
            
            # Call the ChatGPT API
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Extract the generated text
            generated_text = response.choices[0].text.strip()
            results.append({"review": review, "suggestion": generated_text})

        return {"results": results}

api.add_resource(ChatGPT, '/webhook')

if __name__ == '__main__':
    app.run(debug=True)
