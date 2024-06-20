from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.qdrant import QdrantTranslator
from langchain.prompts import ChatPromptTemplate

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,

)

import streamlit as st

import re
import csv





embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

csv_file = 'data.csv'
question = []   
answer = []
issue = []
os = []
with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        question.append(row[0])
        answer.append(row[1])
        issue.append(row[2])
        os.append(row[3])

from langchain_core.documents import Document
docs = []

for i in range(0,len(question)):
    docs.append(Document(
        page_content=f"question:{question[i]}, answer: {answer[i]}",
        metadata={"issue":issue[i], "os": os[i]}
        )
    )



vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="it_sol_chatbot",
)

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",groq_api_key="YOUR_API_KEY")

metadata_field_info = [
    AttributeInfo(
        name="issue",
        description="The issue in the computer",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="os",
        description="The operating system in the computer",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="solution",
        description="the solution for the issue",
        type="string or list[string]",
    ),
   
]

document_content_description = "Brief description of the issue in computer"


prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)

output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=QdrantTranslator(metadata_key="metadata"),
    k=1
)

data = retriever.invoke("My computer is overheating then usual")
print(data)







def get_bot_response(question):
    data = retriever.invoke(question)
    match = re.match(r'question:(.*?),\s*answer:(.*)', data[0].page_content, re.IGNORECASE)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
    system = "You are a helpful IT support assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm
    ans = chain.invoke({"text": f"""This is the question asked {question} and this is the answer to it found in database {answer} Please show a formatted answer for this inabout 30 words"""}).content
    return f"Suggestion: {ans}"


st.set_page_config(page_title="IT Support Chatbot", page_icon=":robot_face:")

# Add a header with a logo
st.markdown("""
<div style="text-align: center;">
    <img src="https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg?t=st=1718820798~exp=1718824398~hmac=d55ab3e2694326371382a11ea2e6e19a2165bd9280d28df9998921c3c477ef90&w=740" alt="Chatbot Logo" width="100"/>
    <h1 style="color: #0078D7;">IT Support Chatbot</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; font-size: 18px; color: #555;">
    Welcome to the IT Support Chatbot! This bot can help you troubleshoot common computer issues and provide IT solutions.
</p>
""", unsafe_allow_html=True)

# Add a divider
st.markdown("<hr/>", unsafe_allow_html=True)

# Text input for user query
user_query = st.text_input("Enter your question:", placeholder="E.g., Why is my computer overheating?")

if st.button("Ask"):
    # Call the backend or NLP model function
    bot_response = get_bot_response(user_query)
    
    # Display bot's response
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <h4 style="color: #0078D7;">Bot's Response:</h4>
        <p style="color: #333;">{bot_response}</p>
    </div>
    """, unsafe_allow_html=True)








    


