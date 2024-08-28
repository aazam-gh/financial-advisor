import streamlit as st
import pandas as pd
import google.generativeai as gemini_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Streamlit UI
st.title("Q&A Retrieval System")

# Load the CSV file
csv_path = '/mnt/data/mfInstruments.csv'
df = pd.read_csv(csv_path)

# Display the first few rows of the dataframe
st.write("Loaded data:")
st.write(df.head())

# Assuming the CSV has columns like 'question' and 'answer'
texts = df.apply(lambda row: f"Q: {row['question']} A: {row['answer']}", axis=1).tolist()

# Configure the Gemini client with your API key
GOOGLE_API_KEY = st.text_input("Enter your Google API key:", type="password")
if GOOGLE_API_KEY:
    gemini_client.configure(api_key=GOOGLE_API_KEY)

    # Embed the texts
    results = [
        gemini_client.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Qdrant x Gemini",
        )
        for text in texts
    ]

    # Set up Qdrant client and create a collection
    collection_name = "faq_collection"
    client = QdrantClient(url="http://localhost:6333")

    # Assuming the embeddings are 768-dimensional
    if not client.get_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            )
        )

    # Prepare and insert points into the Qdrant collection
    points = [
        PointStruct(
            id=idx,
            vector=result['embedding'],
            payload={"text": text},
        )
        for idx, (result, text) in enumerate(zip(results, texts))
    ]
    client.upsert(collection_name=collection_name, points=points)

    # User input for query
    query = st.text_input("Enter your query:")
    if query:
        query_vector = gemini_client.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query",
        )["embedding"]

        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
        )

        # Display the most relevant result
        st.write("Search Results:")
        st.write(search_results)
