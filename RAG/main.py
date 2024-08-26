import os
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure Google API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load and read the CSV file
def load_csv_data(csv_path):
    data = pd.read_csv(csv_path)
    return data

# Function to split the CSV text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and return a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Generate embeddings for the text chunks
    embedding_vectors = [embeddings.embed(text) for text in text_chunks]
    
    # Debugging: Check if embeddings are generated
    if not embedding_vectors or not embedding_vectors[0]:
        raise ValueError("Embeddings are empty or not generated correctly.")
    
    # Create and return the FAISS vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_vectors)
    return vector_store

# Function to create a conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as comprehensively as possible based on the provided context, considering the user's financial goals, risk tolerance, and current situation. Make sure to provide all relevant details and avoid providing incorrect or misleading information. If the answer is not available in the provided context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate a response
def user_input(user_question, user_profile, mf_data):
    # Convert DataFrame to text and split into chunks
    text = mf_data.to_string()
    text_chunks = get_text_chunks(text)

    if len(text_chunks) == 0:
        st.error("No valid text chunks were generated from the mutual fund data.")
        return

    try:
        vector_store = get_vector_store(text_chunks)  # Rebuild the FAISS index
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question, "user_profile": user_profile}
        )

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main function to run the Streamlit app
def main():
    st.set_page_config("Personalized Financial Advisor")
    st.header("Personalized Financial Advisor")
    
    # Correct path using raw string
    csv_path = r"C:\Users\USER\OneDrive\Desktop\RAG\mfInstruments.csv"
    mf_data = load_csv_data(csv_path)

    user_question = st.text_input("Ask a Question")
    user_profile = st.text_input("Enter your financial goals, risk tolerance, and current situation")

    if user_question and user_profile:
        user_input(user_question, user_profile, mf_data)

if __name__ == "__main__":
    main()
