import os
import asyncio
import aiofiles # type: ignore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_DATASETS_CACHE"] = "/tmp"

# The embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# The vector database to use to store the vectors
# Note: Chroma's add_documents is synchronous.
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="/tmp/chroma_db",
)


# The agent class with corrected async methods
class Agent:
    def __init__(self):
        print("Initializing TestAgent...")

        # The language model to use for generating responses
        self.llm = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b",
            temperature=0.2
        )

        # Re-ranker model
        self.re_ranker = FlashrankRerank(top_n=3)

    # The ingestion pipeline
    async def ingest(self, file_name: str):
        print(f"Ingesting file: {file_name}")
        async with aiofiles.open(file_name, "r", encoding="utf-8") as file:
            file_content = await file.read()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts_chunks = text_splitter.create_documents([file_content])
        
        # Asynchronously add documents to the vector store
        ids = await asyncio.create_task(vector_store.aadd_documents(texts_chunks))
        return ids
    
    # The querying pipeline
    async def query(self, query: str):
        print(f"Running query: '{query}'")
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Multi-query retriever
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=self.llm

        )

        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=self.re_ranker,
            base_retriever=multi_retriever
        )

        docs = await compressed_retriever.ainvoke(query)
        print(f"Docs retrieved for question - {query} : {docs}")
    
        return [
        {
            "page_content": doc.page_content,
            "metadata": {
                "id": doc.metadata.get("id", None),
            },
        }
        for doc in docs
    ]

    async def get_response(self, context_list: list, messages : list) -> str:
        """
        This method generates a summary response using a list of tool messages as context.
        """

        # print(f"Past messages available :- {messages}")

        template = """
        You are a travel agent assistant.
        when giving a final response:
            1. Go into a thought process internally.
            2. Do NOT display your thought process.
            3. Only output the final answer.
            4. The final output would appear to address all the questions involved as a single input.

        Your job is to use the answer you decided for each question during the thought process and give one summary response for all the questions.

        The "Past_Chats" towards the end of the prompt is a list(python list) of past conversations you've had with the user, use it as your memory, it would be empty if you've not had any past conversations with the user.
        The system has seperated the different questions in the user's input, generated context for each question and passed it to the "Input_List" (a list containing python dictionaries, each dictionary contains a question and its context) at the end of the prompt.

        The thought process involves checking out each question, checking out their corresponding context, using the memory provided to you, decide if any of the following listed below is true.
        1. The context attached to the question provides an accurate information for the question.
        2. The question is related in a way or is contained in the past conversations you've had with the user.
        3. The question is a friendly conversation like greetings, exchangine pleasantries, talking about the user's mood or directly related to travels.

        if any or all of the above is true then you can decide an answer for the question, if none of them is true, then the question would get a response indicating you don't have an answer for it.

        Past_Chats : {messages}
        Input_List: {context}
        Response:"""

        # 1. Create the full chain using the template object itself
        rag_chain = (
            ChatPromptTemplate.from_template(template)
            | self.llm
            | StrOutputParser()
        )

        # 2. Invoke the chain with the required input variables in a dictionary
        response = await rag_chain.ainvoke({"context": context_list, "messages" : messages})
        # print(f"response {response}")

        return AIMessage(content=response)
    
agent = Agent()
