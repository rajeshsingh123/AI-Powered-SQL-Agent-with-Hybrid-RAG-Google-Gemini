import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import tracing_v2_enabled  # Use LangChain's tracing
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage
import sys
from sqlalchemy import create_engine, text
import pinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.vectorstores import Pinecone as PineconeVectorStore  # Import PineconeVectorStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI



os.environ["PINECONE_API_KEY"] = "pcsk_2v1zKw_72pzGASwxmNVRG35xzJis5nQavr7H1LiY7tbaR5v9R6QaiowhFzvZw28Q2jQubw"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDdYFxGsK9FD_VHDA5mhXeRae24vQ2HS1Q"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] ="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"  # Replace with your actual API key
os.environ["LANGCHAIN_PROJECT"] = "GenAiProject"

print(os.environ.get("LANGCHAIN_TRACING_V2"))
print(os.environ.get("LANGCHAIN_API_KEY"))
print(os.getenv("LANGCHAIN_PROJECT"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


user = 'username'
password = 'password'
host = 'hostname' 
database = 'database name'

# 🔹 Define Paths
SCHEMA_FILE = r"D:\Django_p\Chatbot\nlptosql_project\nlptosql_project\TablesInfo.txt"
INDEX_NAME = "hybrid-rag-index"

# 🔹 Initialize Pinecone Client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 🔹 Function to Load Schema
def load_schema(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 🔹 Function to Split Text
def split_text(schema_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=50)
    return text_splitter.split_text(schema_text)

# 🔹 Function to Create Pinecone Index
def create_pinecone_index(index_name, dimension=768):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,  # GoogleGenerativeAIEmbeddings uses 768-dim vectors
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

# 🔹 Function to Insert Data into Pinecone
def insert_into_pinecone(index, chunks, embedding_model):
    for i, chunk in enumerate(chunks):
        embedding_vector = embedding_model.embed_query(chunk)
        index.upsert(
            vectors=[{"id": f"doc_{i}", "values": embedding_vector, "metadata": {"text": chunk}}]
        )
    print("✅ Data inserted into Pinecone successfully.")

# 🔹 Function to Create Hybrid Retriever (Pinecone + BM25)
def create_hybrid_retriever(index, chunks, embedding_model):
    # 🔹 Load BM25 Retriever
    bm25_retriever = BM25Retriever.from_texts(chunks, k=5)

    # 🔹 Define Pinecone Retriever using PineconeVectorStore
    vectorstore = PineconeVectorStore(index, embedding_model, "text")  # Corrected line
    pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 🔹 Combine Both in an Ensemble Retriever
    return EnsembleRetriever(retrievers=[pinecone_retriever, bm25_retriever], weights=[0.5, 0.5])


# def create_hybrid_retriever(index, schema_text, embedding_model):
#     """
#     Creates a hybrid retriever using Pinecone and BM25 with the entire schema_text.

#     Args:
#         index: Pinecone index object.
#         schema_text: The entire schema text as a single string.
#         embedding_model: Embedding model for Pinecone.

#     Returns:
#         EnsembleRetriever: A hybrid retriever combining Pinecone and BM25.
#     """

#     # 🔹 Load BM25 Retriever with the entire schema_text as a single document
#     bm25_retriever = BM25Retriever.from_texts([schema_text], k=5)  # Note the [schema_text]

#     # 🔹 Define Pinecone Retriever using PineconeVectorStore
#     vectorstore = PineconeVectorStore(index, embedding_model, "text")
#     pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     # 🔹 Combine Both in an Ensemble Retriever
#     return EnsembleRetriever(retrievers=[pinecone_retriever, bm25_retriever], weights=[0.5, 0.5])
# 🔹 Main Pipeline Execution

schema_text = load_schema(SCHEMA_FILE)
chunks = split_text(schema_text)

# Initialize Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create or Load Pinecone Index
pinecone_index = create_pinecone_index(INDEX_NAME)

# Insert Data into Pinecone (Only if Fresh Index)
if pinecone_index.describe_index_stats()["total_vector_count"] == 0:
    print("Creating and inserting data into Pinecone...")
    insert_into_pinecone(pinecone_index, chunks, embedding_model)
else:
    print("✅ Pinecone Index already populated.")

hybrid_retriever = create_hybrid_retriever(pinecone_index, chunks, embedding_model)


# hybrid_retriever = create_hybrid_retriever(pinecone_index, schema_text, embedding_model)


def reformulate_question(latest_question, chat_history, llm):
    # Create a prompt template for reformulating the question
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "latest_question"],
        template="""
        Given the chat history and the latest user question, 
        reformulate the question so that it is a complete, standalone question.
        that can be understood without any context from the chat history. 
        If the latest question is unrelated to the previous conversation or 
        does not require any historical context to be understood, 
        return it exactly as it is. Do NOT answer the question.
        Example-If user has ask previously that what is the bill date of a/c no 1234.
        And Now he is asking what is its due date.Then you should reformulate it like what is the due date of a/c no 1234.

        Chat History: {chat_history}
        Latest Question: {latest_question}
        Reformulated Question:
        """
    )
    
    # Create an LLM Chain for reformulating the question
    reformulate_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Format chat history into a readable format for the prompt
    # formatted_history = "\n".join([
    #     f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
    #     for msg in chat_history
    # ])
    formatted_history = "\n---\n".join([
    f"User: {msg['content']}" if msg['role'] == 'user' else f"AI: {msg['content']}"
    for msg in chat_history
    ])
    
    
    # Invoke the chain to reformulate the question
    reformulated_question = reformulate_chain.run({
        "chat_history": formatted_history,
        "latest_question": latest_question
    }).strip()
    
    return reformulated_question


def generate_sql(latest_question, chat_history, retriever, llm):
    # Step 1: Reformulate the latest question (if needed)
    if len(chat_history)>0:
        latest_question = reformulate_question(latest_question, chat_history, llm)
    print(latest_question,"-------------------------------")
    reformulated_query=latest_question
    # Step 2: Get relevant documents from retriever
    retrieved_docs = retriever.get_relevant_documents(reformulated_query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Step 3: Format the chat history for better context
    # formatted_history = "\n---\n".join([
    #     f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
    #     for msg in chat_history
    # ])
    formatted_history = "\n---\n".join([
    f"User: {msg['content']}" if msg['role'] == 'user' else f"AI: {msg['content']}"
    for msg in chat_history
    ])
    
    # Step 4: Create a new prompt template with improved context handling
    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "latest_question"],
        template="""        
        Given the following database schema:
        {context}
        
        User's conversation history is provided to maintain context. 
        If the latest question is related to the previous questions, use the history to understand the full context.
        If it is a new and independent question, focus only on the latest question.
        
        Conversation History:
        {chat_history}
        
        Latest Question:
        {latest_question}
        
        Provide only the MS SQL query for the question, without any explanation, comments, or code block markers.
        Ensure that the query limits the results to a maximum of 20 rows.
        Use the appropriate SQL syntax for limiting rows, as I am using MS SQL.
        Output should be plain text without any formatting.
        Do not generate any UPDATE, INSERT, or DELETE queries. Only SELECT queries are allowed.
        
        SQL Query:
        """,
    )
    
    # Debug: Print the final prompt going for SQL generation
    # print("\n=== Final Prompt for SQL Generation ===")
    # print(f"Context:\n{context}")
    # print(f"Chat History:\n{formatted_history}")
    # print(f"Latest Question:\n{latest_question}")
    # print("======================================\n")
    
    # Step 5: Use LLMChain directly instead of RetrievalQA
    sql_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    
    # Step 6: Pass all inputs explicitly
    response = sql_chain.run({
        "context": context,
        "chat_history": formatted_history,
        "latest_question": latest_question
    }).strip()
    
    return response


def validate_and_rewrite_sql(user_query, retriever, llm, generated_query, retry_on_error=False,Error=None):
    """
    Validate and optionally rewrite the generated MS SQL query using an LLM.
    
    Parameters:
        user_query (str): The original user question.
        retriever: The retriever used for getting relevant documents.
        llm: The language model used to validate and rewrite SQL queries.
        generated_query (str): The initially generated SQL query.
        retry_on_error (bool): If True, retry rewriting the query on error or no table retrieval.
    
        
    Returns:
        tuple: (is_valid, revised_query)
               is_valid (bool): True if the query is valid, False otherwise.
               revised_query (str): The revised or validated query.
    """
    
    def check_query_with_llm(query):
        """Use LLM to check if the query is valid and free of forbidden statements."""
        validation_prompt = f"""
        Check the following MS SQL query for correctness:
        - Ensure that the query is syntactically correct for MS SQL.
        - Confirm that it does not contain any INSERT, UPDATE, or DELETE statements.
        - Confirm that it follows best practices and limits the results to a maximum of 20 rows.
        - Respond with 'Valid' if the query is correct and safe to run.
        - Otherwise, explain the issue briefly.
        
        SQL Query:
        {query}
        
        Result:
        """
        
        # Using the LLM to validate the query
        response = llm.invoke(validation_prompt)
        return response.content.strip()
    
    def rewrite_query():
        """Rewrite the query by retrieving documents again and using the LLM."""
        retrieved_docs = retriever.get_relevant_documents(user_query)
        print("Documents Retrieved for Rewriting:", len(retrieved_docs))
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        print("Context for Rewriting:", context)
        
        prompt_template = PromptTemplate(
            input_variables=["context","question"],
            template="""
            Given the following database schema:
            {context}
            
            Provide only the MS SQL query for the following question, without any explanation, comments, or code block markers:
            - Ensure that the query limits the results to a maximum of 20 rows, regardless of how many rows the query would normally return.
            - Use the appropriate SQL syntax for limiting rows, as I am using MS SQL.
            - Output should be plain text without any formatting, special characters, or labels.
            - Do not generate any UPDATE, INSERT, or DELETE queries. Only SELECT queries are allowed.


            So Consider this  previous generated  error and query before generating new query.If its None Ignore. 
            
            User Question: {question}
            SQL Query:
            """,
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt_template})
        return qa_chain.run(user_query)
    
    # Step 1: Validate the generated query using LLM
    validation_result = check_query_with_llm(generated_query)
    print(validation_result)
    if validation_result == "Valid":
        print("Query is valid.")
        return True, generated_query
    else:
        print("Validation Issue:", validation_result)
        if retry_on_error:
            print("Retrying with rewritten query...")
            return True, rewrite_query()
        else:
            return False, generated_query
    
    # If no issues, return True and the validated query
    return True, generated_query

def execute_query_mssql(sql_query, user, password, host, database, driver="ODBC Driver 17 for SQL Server"):
    """
    Executes the given SQL query on the specified MSSQL database.

    Parameters:
    - sql_query (str): The SQL query to execute.
    - user (str): The MSSQL username.
    - password (str): The MSSQL password.
    - host (str): The MSSQL host address (e.g., 'localhost' or an IP address).
    - database (str): The name of the database to use.
    - driver (str): The ODBC driver to use. Default is 'ODBC Driver 17 for SQL Server'.

    Returns:
    - list: The result of the query as a list of rows.
    """
    try:
        # Create the connection string for MSSQL
        connection_string = f"mssql+pyodbc://{user}:{password}@{host}/{database}?driver={driver}"
        # Create the engine
        engine = create_engine(connection_string)

        # Connect to the database and execute the query
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            data = result.fetchall()
            connection.close()

        return data,None

    except Exception as e:
        print("An error occurred:", e)
        return None,e

def SqlToAnalysis(table_results, user_query, llm):
    # Validate inputs
    print("table",table_results)
    print("user query",user_query)
    if not table_results or not user_query:
        print("Table results or user query is empty. Returning None.")
        return None
    
    # Improved prompt for better context and results
    prompt = f"""
    You are an expert data analyst.
    
    Below is the table extracted based on the user's question.
    Your task is to analyze the table and provide a clear, concise answer to the user's question.
    
    User Question: {user_query}
    
    Table Extraction: 
    {table_results}
    
    Analyze the above table and return the answer in text format.
    - Ensure the answer is only based on the information present in the table.
    - If the extracted table is not related to the user's question or does not contain enough information, return: None
    - Do not generate any extra text or explanations.
    
    Answer:
    """
    # print(prompt)
    try:
        response = llm.invoke(prompt)
        # Ensure response is not empty or irrelevant
        print("Response",response)
        if response.content.strip().lower() in ['none', 'null', '']:
            return None
        return response.content.strip()
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None
import time
from typing import List, Dict, Tuple


def check_sql_usage(chat_history: List[Dict[str, str]], user_query: str, llm) -> Tuple[bool, str]:
    """
    Determines whether the user's question requires an SQL query.
    Also checks if the query references previous history correctly.
    
    Parameters:
    - chat_history: List of dictionaries containing previous user and bot messages.
    - user_query: The latest user question.
    - llm: Predefined LLM model for validation.
    
    Returns:
    - (bool, str):
        - True if SQL DB should be used, False otherwise.
        - The response if SQL is not needed.
    """
    prompt = f"""
    You are an AI assistant helping to determine whether a user's question requires querying an SQL database.
    You only support SELECT queries; INSERT, UPDATE, DELETE, or DDL queries are not supported.
    
    Conversation history:
    {chat_history}
    
    User's latest query:
    "{user_query}"
    
    Instructions:
    1. If the user's query is unrelated to SQL (e.g., "How are you?"), respond appropriately without using SQL.
    2. If the user references previous history but no valid context exists (e.g., asks "What's its due date?" without prior account details), return an error like "I don't have enough history."
    3. If the query is about inserting, updating, or deleting data, return "Database modification is not allowed."
    4. If user is asking questions based on history (by using past verbs like that,was,are ) and you do not have proper histroy then return Sorry i do not have proper history.
    5. Otherwise, return "Use SQL."
    """
    
    response = llm.invoke(prompt)
    print(response.content)
    if "Use SQL" in response.content:
        return True, ""
    else:
        return False, response

def process_user_query(user_query, chat_history):
    try:
        """Handles user query processing, SQL generation, validation, execution, and response generation."""
        with tracing_v2_enabled(project_name="GenAiProject"):
            
            use_sql,reply=check_sql_usage(chat_history,user_query,llm)
            if use_sql:
                print("We are using sql generation for this")
            else:
                print(reply.content)
                return reply.content
            sql_query = generate_sql(user_query, chat_history, hybrid_retriever, llm)
            print("Userquery",user_query)
            print("User q",user_query)
            print(chat_history)
            # for i in range(0,3):
            #     table_results, error = execute_query_mssql(sql_query, user, password, host, database)
            # # table_results='21-02-2024'
            #     print("table result",table_results)
            #     if table_results is None and error is not None:
            #         print("Error during query execution:", error)
            #         if i==2:
            #             return f"Query failed again with error: {error}"
            #         bool_val, sql_query = validate_and_rewrite_sql(user_query, hybrid_retriever, llm, sql_query, retry_on_error=True, Error=error)                        
                    
            #     else:
            #         response = SqlToAnalysis(table_results, user_query, llm)
            #         chat_history.extend([
            #             {"role": "user", "content": user_query},  # Store as dict
            #             {"role": "bot", "content": sql_query}     # Store as dict
            #         ])

            return sql_query
            
    except Exception as ex:       
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_tb.tb_lineno)
        print("Error in fun is ",ex,exc_tb.tb_lineno)
        return f"An error occurred: {str(ex)}"
