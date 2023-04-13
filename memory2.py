import openai
import pinecone
import time
import os
import subprocess
from collections import deque
from config import Config, Singleton

cfg = Config()


class PineconeMemory(metaclass=Singleton):
    def __init__(self):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "oaie"
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        self.vec_num += 1
        return f"Inserting data into memory at index: {self.vec_num - 1}:\n data: {data}"

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


memory = PineconeMemory()
memory.clear()
full_message_history = []


def openai_call(prompt, model=cfg.OPENAI_API_MODEL, temperature=0.5, max_tokens=100):
    while True:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError:
            time.sleep(10)
        else:
            break


def main_agent():
    # Define the product objective
    objective = "The objective of the product is to ..."

    # The Agent should be able to answer questions about the product.
    question = "What is the main purpose of the product?"
    answer = openai_call(f"Objective: {objective}\nQuestion: {question}\nAnswer:")
    print(answer)

    # Read reviews from CSV file
    # reviews = ...
    # Extract and preprocess information from reviews
    # extracted_information = ...

    # Store extracted information in memory
    for info in extracted_information:
        memory.add(info)

    # Retrieve relevant information from memory
    relevant_info = memory.get("some_query")

    # The Agent should be able to write a report about the product
    report_prompt = f"Objective: {objective}\nRelevant information: {relevant_info}\nWrite a report about the product:"
    report = openai_call(report_prompt)
    print(report)





if __name__== "__main__":
    main_agent()

# The following code block is for demonstration purposes.
# You should replace it with your actual CSV reading and data extraction functions.

# Sample reviews
reviews = [
    "I love this product! It's so easy to use and works as expected.",
    "The product is well-designed, but I had issues with the customer support.",
    "This product has changed my life! I can't imagine living without it.",
]

# Sample extracted information
extracted_information = [
    "positive feedback on ease of use",
    "well-designed product",
    "negative feedback on customer support",
    "life-changing product",
]

# Function to read reviews from a CSV file (replace with your actual implementation)
def read_reviews_from_csv(file_path):
    # Read and parse CSV file
    # Return a list of review strings
    pass

# Function to extract information from reviews (replace with your actual implementation)
def extract_information_from_reviews(reviews):
    # Process and analyze reviews
    # Return a list of extracted information strings
    pass

# Update the main_agent function to use the new functions
def main_agent():
    # Define the product objective
    objective = "The objective of the product is to ..."

    # The Agent should be able to answer questions about the product.
    question = "What is the main purpose of the product?"
    answer = openai_call(f"Objective: {objective}\nQuestion: {question}\nAnswer:")
    print(answer)

    # Read reviews from CSV file
    reviews = read_reviews_from_csv("path/to/your/csv_file.csv")
    # Extract and preprocess information from reviews
    extracted_information = extract_information_from_reviews(reviews)

    # Store extracted information in memory
    for info in extracted_information:
        memory.add(info)

    # Retrieve relevant information from memory
    relevant_info = memory.get("some_query")

    # The Agent should be able to write a report about the product
    report_prompt = f"Objective: {objective}\nRelevant information: {relevant_info}\nWrite a report about the product:"
    report = openai_call(report_prompt)
    print(report)




if __name__ == "__main__":
    main_agent()




import time
import openai
from dotenv import load_dotenv
from config import Config
import token_counter
from memory import BrainMemoryCreator, BrainMemoryRetriever

cfg = Config()
openai.api_key = cfg.openai_api_key

# Create instances of BrainMemoryCreator and BrainMemoryRetriever
memory_creator = BrainMemoryCreator()
memory_retriever = BrainMemoryRetriever()

# ... (rest of the code) ...

def chat_with_ai(prompt, user_input, full_message_history, permanent_memory, token_limit):
    while True:
        try:
            # Retrieve the language model from a configuration object
            model = cfg.fast_llm_model

            # Calculate the maximum number of tokens that can be sent to the model
            send_token_limit = token_limit - 1000

            # Retrieve the relevant messages from memory based on the last 5 messages in the conversation
            relevant_memory = memory_retriever.get(str(full_message_history[-5:]))

            # Print out memory statistics
            print('Memory Stats: ', permanent_memory.get_stats())

            # Generate a context for the conversation, including the prompt, relevant messages, and user input
            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model)

            # Ensure that the context contains no more than 2500 tokens
            while current_tokens_used > 2500:
                # Remove the oldest message from relevant memory
                relevant_memory = relevant_memory[1:]
                # Regenerate the context with the updated relevant memory
                next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                    prompt, relevant_memory, full_message_history, model)

            # Calculate the number of tokens used by adding the token count of the user's input to the current token count
            current_tokens_used += token_counter.count_message_tokens([create_chat_message("user", user_input)], model)


            # Add messages from full message history to the current context until the maximum token limit is reached or there are no more messages to add
            while next_message_to_add_index >= 0:
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = token_counter.count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                current_context.insert(insertion_index, full_message_history[next_message_to_add_index])

                current_tokens_used += tokens_to_add

                next_message_to_add_index -= 1

            # Add the user's input to the current context
            current_context.extend([create_chat_message("user", user_input)])

            # Calculate the number of tokens remaining for the model to use
            tokens_remaining = token_limit - current_tokens_used

            # Print out each message in the current context except for the prompt message
            for message in current_context:
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                print(f"{message['role'].capitalize()}: {message['content']}")
                print()

            # Generate a response from the model using the current context and remaining tokens
            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Add the user's input and the model-generated response to the message history
            full_message_history.append(create_chat_message("user", user_input))
            full_message_history.append(create_chat_message("assistant", assistant_reply))

            # Add the model-generated response to permanent memory
            memory_creator.add(assistant_reply)

            # Return the model-generated response
            return assistant_reply

        except openai.error.RateLimitError:
            # Wait for 10 seconds if the API rate limit is reached
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)
