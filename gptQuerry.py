import openai
import os
from typing import Dict

def generate_responses(review_dict: Dict[int, str], prompts: [str], max_tokens: int = 1000) -> Dict[int, Dict[str, str]]:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    results = {}

    for review_index, review_text in review_dict.items():
        results[review_index] = {}
        for prompt in prompts:
            input_text = f"{prompt}\n\n{review_text}"
            response = openai.Completion.create(
                engine="gpt-3.5-turbo",
                prompt=input_text,
                max_tokens=max_tokens,pip install langflow
                n=1,
                stop=None,
                temperature=0.7,
            )
            results[review_index][prompt] = response.choices[0].text.strip()

    return results



"""
I have a project folder.
I have several .py files.

One of the is app.py, that runs a flask app.

I have a dataPrep.py that is in charge with receiving a csv and transforming it ito a dictionary.

I have a gptQuerry.py that is in charge with reciving a dictionary that has reviews and transforming it into a disctionary that has reviews, questions and answers from gpt.

How do I do so that I can call these files into app.py?"""





"""
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


"""