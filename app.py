import openai
import pandas as pd
from io import StringIO
from flask import Flask, request
from flask_restful import Resource, Api

# Replace with your OpenAI API key
openai.api_key = "your-api-key"

app = Flask(__name__)
api = Api(app)

class ChatGPT(Resource):
    def post(self):
        csv_data = request.data.decode("utf-8")
        df = pd.read_csv(StringIO(csv_data))

        # Process the CSV data and generate a prompt for ChatGPT
        prompt = generate_chatgpt_prompt(df)

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

        return {"response": generated_text}

def generate_chatgpt_prompt(df):
    # Process the DataFrame to create a prompt for ChatGPT
    # For example, you can aggregate the reviews and find common issues or areas of improvement
    # In this example, we simply pass the raw CSV data to ChatGPT
    return f"Please analyze the following product reviews and suggest ways to improve the product:\n\n{df.to_string()}"

api.add_resource(ChatGPT, '/webhook')

if __name__ == '__main__':
    app.run(debug=True)
