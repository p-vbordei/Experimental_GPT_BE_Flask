
"""

I need a backend application that will receive 1000 reviews of a Amazon Product. 
Backend is running on a Flask App. 
Frontend is on Wix.
Communication is done via webhooks with ngrok,
Development on a M1 Macbook.
I only know python. 


The way the code should work is the following:


The reviews are read and as much information as possible is extracted from each reviews. 
The extracted information is then stored in a database, after having been cleaned and preprocessed, as well as tagged for relevant information.
Relevant information is any information that will be used to help the model improve the physhical product.



This application should do the following:


Use Chat GPT. The name of the system that uses chat GPT for this will be The Agent.
The Agent should have acces to memory. It can recall information from memory and can write information to memory.
Memory should be short term and long term, meaning that both small, task level information should be saved and long term information should be saved, as required.

Long term information can be retreived to be used as context for The Agent

The Agent should be able to answer questions about the product.

The Agent should be independent, meaning that no user input is expected, but the Agent should be able to ask questions to the user, if required.

The Agent should be able to write a report about the product, based on the information it has gathered.


"""

##################### THIS PART OF THE APPLICATION DEFINES MEMORY FUNCTIONS ########################


# Reviews Input = data from a csv file


# Credit: AutoGPT Codebase
from config import Config, Singleton
import pinecone
import openai


cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class PineconeMemory(metaclass=Singleton):
    def __init__(self):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "oaie"
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with memory.
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()


# Initialize memory and make sure it is empty.
# this is particularly important for indexing and referencing pinecone memory
memory = PineconeMemory()
memory.clear()
full_message_history = []




# Function to query records from the Pinecone index
def query_records(index, query, top_k=1000):
    results = index.query(query, top_k=top_k, include_metadata=True)
    return [f"{task.metadata['task']}:\n{task.metadata['result']}\n------------------" for task in results.matches]

# Get embedding for the text
def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query Pinecone index using a string.")
    parser.add_argument('objective', nargs='*', metavar='<objective>', help='''
    main objective description. Doesn\'t need to be quoted.
    if not specified, get objective from environment.
    ''', default=[os.getenv("OBJECTIVE", "")])
    args = parser.parse_args()

    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY)

    # Connect to the objective index
    index = pinecone.Index(PINECONE_TABLE_NAME)

    # Query records from the index
    query = get_ada_embedding(' '.join(args.objective).strip())
    retrieved_tasks = query_records(index, query)
    for r in retrieved_tasks:
        print(r)

if __name__ == "__main__":
    main()





# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")

# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(
        table_name, dimension=dimension, metric=metric, pod_type=pod_type
    )

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
                # Use completion API
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
            else:
                # Use chat completion API
                messages = [{"role": "system", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):
    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective},
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
    Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    
    context = context_agent(query=objective, n=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    return openai_call(prompt, temperature=0.7, max_tokens=2000)


def context_agent(query: str, n: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        n (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    query_embedding = get_ada_embedding(query)
    results = index.query(query_embedding, top_k=n, include_metadata=True, namespace=OBJECTIVE)
    # print("***** RESULTS *****")
    # print(results)
    sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata["task"])) for item in sorted_results]


# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        index.upsert(
            [(result_id, vector, {"task": task["task_name"], "result": result})],
	    namespace=OBJECTIVE
        )

        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

    time.sleep(1)  # Sleep before checking the task list again



from langchain.tools import BaseTool

class BrainMemoryCreator(BaseTool):
    name = "Brain Memory Creator"
    description = '''
    You have a brain that can remember things and retrieve when needed. This tool is for you to create a memory.
    
    Your memory will be storage as a key-value pair. 
    
    The input to this tool should be a comma separated list of key and value, like "key, value"
    
    For example, "Birthday, 2023/1/1" means to remember birthday is 2023/1/1
    '''

    def _run(self, tool_input: str) -> str:
        console.log(f"[bold green]call {self.name}[/]: {tool_input}")
        key, value = tool_input.split(",")
        try:
            brain = Brain()
            brain.read()
            brain.append(key, value)
            brain.save()
            r = "Memory Success"
        except Exception as e:
            r = f"Error: {e}"
        console.log(f"[bold green]return:[/]\n {r}")
        return r

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("BrainMemoryCreator does not support async")


class BrainMemoryRetriever(BaseTool):
    name = "Brain Memory Retriever"
    description = '''
    You have a brain that can remember things and retrieve when needed. This tool is for you to retrieve from memory.
    
    Your memory will be storage as a key-value pair. Input a key so that you can get the value.
    
    The input to this tool should be the key as a string
    
    For example, "Birthday"
    '''

    def _run(self, tool_input: str) -> str:
        console.log(f"[bold green]call {self.name}[/]: {tool_input}")
        try:
            brain = Brain()
            brain.read()
            r = brain.get(tool_input)
        except Exception as e:
            r = f"Error: {e}"
        console.log(f"[bold green]return:[/]\n {r}")
        return r

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("BrainMemoryRetriever does not support async")





















##################### THIS PART OF THE APPLICATION TRIES TO USE MEMORY ########################



import pandas as pd
import numpy as np
import promptlayer

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chat_models import PromptLayerChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from rich.console import Console
from rich.table import Table
console = Console()

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
WOLFRAM_ALPHA_APPID = os.getenv('WOLFRAM_ALPHA_APPID')
PROMPTLAYER_API_KEY = os.getenv('PROMPTLAYER_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')



from memory import PineconeMemory


# Initialize memory and make sure it is empty.
# this is particularly important for indexing and referencing pinecone memory

memory = PineconeMemory()


print('Using memory of type: ' + memory.__class__.__name__)


# Initialize the agent
openai = promptlayer.openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


reviews = pd.read_csv('reviews.csv')
querry = {0: "What are the top traits that customers appreciate the most about a product? Provide a description of each trait. ",
          1: "What are the top traits that customers appreciate the least about a product? Provide a description of each trait. ",
          2: "What are the improvements that can be brought to this product? Provide a description of each trait. "}

SystemMessage = """Answer the question based on the context and reviews below. You will answer with bulletpoints and extra clarity as a profesional developer. If the question cannot be answered using the information provided answer with "I don't know".

Context: You are looking at a product sold on amazon.com. We are a competing product development team. Our scope is to better understand the clients need with our product in order to improve the product. """

system_message_prompt = SystemMessagePromptTemplate.from_template(SystemMessage)


HumanMessage = """Question:  {inputQuestion}
                Reviews: {inputReviews} """

human_message_prompt = HumanMessagePromptTemplate.from_template(HumanMessage)

AiMessage = """Answer: """
ai_message_prompt = AIMessagePromptTemplate.from_template(AiMessage)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt, ai_message_prompt])


# Pass the reviews and querry to the promptlayer, get the results and add them to the memory

if len(querry) > 0:
    for i in range(len(reviews)):
        for j in range(len(querry)):
            # PROMPTLAYER
            prompt = chat_prompt.format_prompt(
                inputQuestion=querry[j],
                inputReviews=reviews['review'][i]
            ).to_messages()
            results = chat(prompt) 
            print(results)
       
            full_memory_to_add = f" Human Message: {querry[j]} " \
                f"\n Assistant Reply: {results} " \
                f"\n Reviews: {reviews['review'][i]} "
            memory.add(full_memory_to_add)

            short_memory_to_add = f" Human Question: {querry[j]} " \
                f"\n Assistant Answer: {results} "
            memory.add(short_memory_to_add)

# template_dict = promptlayer.prompts.get("reviewsPromptFeatures_test1")
# https://python.langchain.com/en/latest/modules/models/chat/integrations/promptlayer_chatopenai.html


# memory.get_relevant("Assistant Reply", num_relevant=6)


memory.get_stats()



Knowns = memory.get_relevant("Assistant Answer", num_relevant=12)

from langchain.agents import initialize_agent, load_tools, Tool
from langchain.llms import OpenAI

from langchain import SerpAPIWrapper, LLMChain, LLMMathChain, Wikipedia
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.agents.react.base import DocstoreExplorer

docstore=DocstoreExplorer(Wikipedia())

import json

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

conversationMemory = ConversationBufferMemory(memory_key="chat_history")
search = SerpAPIWrapper()
llm_math = LLMMathChain(llm=llm, verbose=True)
wolfram = WolframAlphaAPIWrapper()

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math.run,
        description="useful for when you need to answer questions about Divide, Multiply, Add and Subtract"
    ),
    Tool(
        name="Wolfram",
        func=wolfram.run,
        description="Useful for when you need to answer questions about math, science, geography, history, politics, arts, business, finance, product materials, material science"
    ),
    Tool(
        name="SearchWiki",
        func=docstore.search,
        description="useful for when you need to ask Wikipedia, a general purpose encyclopedia,  with search"
    ),
    Tool(
        name="LookupWiki",
        func=docstore.lookup,
        description="useful for when you need to ask Wikipedia, a general purpose encyclopedia,  with lookup"
    )
]

QA = [
"Human Message: What are the improvements that can be brought to this product? Provide a description of each trait.  \nAssistant Reply: content='Based on the reviews, it is difficult to identify specific improvements for the product. However, the following observations can be made to help in product development:\\n- The adhesive should be improved so that the gems stick longer to surfaces\\n- The instructions on how to use the device can be improved, making it more user-friendly for anyone who wants to use it\\n- The product descriptions and images need to accurately reflect the product so that there is no confusion when it arrives\\n- The product should be tested to ensure proper functionality and durability before shipping\\n- More refills should be included in the kit to avoid continually buying additional ones' additional_kwargs={} \n",
"Human Message: What are the improvements that can be brought to this product? Provide a description of each trait.  \nAssistant Reply: content='Some potential improvements that can be made to this product based on the reviews are:\\n\\n- Increase the number of gems included or sell larger refill packs\\n- Improve the quality of the plastic components, as some reviewers mentioned that the product feels flimsy or breaks easily\\n- Increase the strength of the adhesive on the gems to ensure they stay in place longer\\n- Improve the design of the machine to prevent gems from getting stuck or falling out during use\\n- Provide clearer instructions or tutorials for how to use the product, especially for younger kids or adults who are not familiar with it\\n- Consider including additional accessories or features that would enhance the user experience, such as different gem shapes or sizes, or the ability to customize the machine with stickers or decorations.' additional_kwargs={} \n"
]


template = """
Assistant is designed to independently be able to assist with creating industrial products, with access to search, Wolfram Alpha, and a calculator. 
Assistent can also summarize information from reviews and generate a report outlining potential product improvements, from answering simple questions to providing 
in-depth explanations and discussions on a wide range of topics.

Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

You have the ability to call external tools. When you call an external tool, it will provide you with some information, which you can use when answering.
Please remember: your answer should be as detailed as possible. If you are going to cite information from the internet, please add the corresponding URL at the appropriate location.

CONTRAINT:  No user assistance

PERFORMANCE EVALUATION:

1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities. 
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below

RESPONSE FORMAT:
{
    "thoughts":
    {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args":{
            "arg name": "value"
        }
    }
}

Ensure the response can be parsed by Python json.loads

GOAL: {human_input}

Historical Information:
"Human Message: What are the improvements that can be brought to this product? Provide a description of each trait.  \nAssistant Reply: content='Based on the reviews, it is difficult to identify specific improvements for the product. However, the following observations can be made to help in product development:\\n- The adhesive should be improved so that the gems stick longer to surfaces\\n- The instructions on how to use the device can be improved, making it more user-friendly for anyone who wants to use it\\n- The product descriptions and images need to accurately reflect the product so that there is no confusion when it arrives\\n- The product should be tested to ensure proper functionality and durability before shipping\\n- More refills should be included in the kit to avoid continually buying additional ones' additional_kwargs={} \n",
"Human Message: What are the improvements that can be brought to this product? Provide a description of each trait.  \nAssistant Reply: content='Some potential improvements that can be made to this product based on the reviews are:\\n\\n- Increase the number of gems included or sell larger refill packs\\n- Improve the quality of the plastic components, as some reviewers mentioned that the product feels flimsy or breaks easily\\n- Increase the strength of the adhesive on the gems to ensure they stay in place longer\\n- Improve the design of the machine to prevent gems from getting stuck or falling out during use\\n- Provide clearer instructions or tutorials for how to use the product, especially for younger kids or adults who are not familiar with it\\n- Consider including additional accessories or features that would enhance the user experience, such as different gem shapes or sizes, or the ability to customize the machine with stickers or decorations.' additional_kwargs={} \n"


Assistant:"""



agent_chain = initialize_agent(tools,
                               llm,
                               agent="conversational-react-description",
                               verbose=True,
                               memory=conversationMemory,
                               template=new_template)



output = agent_chain.run(input=user_input)


while True:
    user_input = input("Human: ")
    if user_input == "exit":
        break
    response = agent_chain.run(input=user_input)
    print(response)