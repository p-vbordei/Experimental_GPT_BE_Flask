
"""

It defines the PineconeMemory class, which is used for storing and retrieving information, 
and two additional classes for creating and retrieving memories: BrainMemoryCreator and BrainMemoryRetriever.
The PineconeMemory class uses Pinecone, a vector database, for storing and querying embeddings. 
The class has methods to add, clear, and retrieve relevant data from memory. 
To convert between text and embeddings, the class uses OpenAI's text-embedding-ada-002 model.
The BrainMemoryCreator and BrainMemoryRetriever classes are tools designed 
to create and retrieve key-value pairs in memory.
 They inherit from the BaseTool class and implement the _run method for synchronous execution.
However, they do not implement the _arun method for asynchronous execution, as they raise a NotImplementedError.
"""


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


import pickle


class Brain:
    def __init__(self):
        self.my_dict = {}

    def save(self):
        with open("brain.pkl", "wb") as f:
            pickle.dump(self.my_dict, f)

    def read(self):
        with open("brain.pkl", "rb") as f:
            self.my_dict = pickle.load(f)

    def append(self, key: str, value: str):
        self.my_dict[key] = value

    def get(self, key: str):
        return self.my_dict[key]


if __name__ == "__main__":
    brain = Brain()
    brain.read()
    print(brain.my_dict)
