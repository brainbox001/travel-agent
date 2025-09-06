# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import CharacterTextSplitter
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.document_compressors import FlashrankRerank
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # 1st step - receiving, embedding and storing the text file in a vector database

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
# )

# # The vector database to use to store the vectors
# vector_store = Chroma(
#     collection_name="my_collection",
#     embedding_function=embeddings,
# )

# with open("./2024_state_of_the_union.txt", "r", encoding="utf-8") as file:
#     state_of_union_content = file.read()


# # Split the text into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# texts_chunks = text_splitter.create_documents([state_of_union_content])

# # Add the chunks to the vector database
# ids = vector_store.add_documents(texts_chunks)

# # Testing step 1 - retrieving the chunks from the vector database to ensure they were stored correctly

# # results = vector_store.similarity_search(
# #     'Who invaded Ukraine?',
# #     k=2
# # )

# # # Print Resulting Chunks
# # for res in results:
# #     print(f"* {res.page_content} [{res.metadata}]\n\n")


# # 1st step ends - 2nd step begins - retrieving the relevant chunks based on the user's query from the vector database and generating an answer using a language model

# # set vector store as the retriever
# retriever =  vector_store.as_retriever(search_kwargs={"k": 3})

# llm = ChatGroq(
#     groq_api_key=os.environ.get("GROQ_API_KEY"),
#     model_name="llama3-8b-8192",
#     temperature=0.2
# )

# # Multi-query retriever
# multi_retriever = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm

# )

# reranker = FlashrankRerank(top_n=3)

# compressed_retriever = ContextualCompressionRetriever(
#     base_compressor=reranker,
#     base_retriever=multi_retriever,
# )

# template = """You are an assistant for question-answering tasks. 
# Use the following retrieved context to answer the question.
# If you have an additional information that is not in the context, you can use it to answer the question.
# If you don't know the answer, say that you don't know.
# Question: {question}
# Context: {context}
# Answer:
# """
# prompt = ChatPromptTemplate.from_template(template)

# # Build the final RAG chain
# rag_chain = (
#     {"context": compressed_retriever, "question": RunnablePassthrough()}
#     | prompt
#     | ChatGroq(temperature=0, model_name="llama3-8b-8192")
#     | StrOutputParser()
# )

# # Invoke the chain to get the final answer
# query = "who recently joined nato"
# answer = rag_chain.invoke(query)
# print(answer)
