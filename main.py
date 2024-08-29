import os
from pathlib import Path
from typing import List, Tuple
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SimilarityPostprocessor
import qdrant_client
import gradio as gr

# Set up paths
base_dir = Path(__file__).parent
dataset_dir = base_dir / "Dataset"
qdrant_path = base_dir / "Finance"

# Ensure the Dataset directory exists
dataset_dir.mkdir(exist_ok=True)

# Load documents
docs = SimpleDirectoryReader(dataset_dir).load_data()

# Initialize SentenceSplitter and process documents
text_parser = SentenceSplitter(chunk_size=1024)
text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(docs):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# Create TextNodes
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(text=text_chunk)
    src_doc = docs[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# Set up Qdrant vector store
client = qdrant_client.QdrantClient(path=str(qdrant_path))
vector_store = QdrantVectorStore(client=client, collection_name="collection")

# Set up embedding model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
    node.embedding = node_embedding

# Set up Gemini LLM
GOOGLE_API_KEY = "AIzaSyDaCWThwHQj5NPaZ0ZEMgFG7U3tqk7-s5E"

# Configure settings
Settings.embed_model = embed_model
Settings.llm = Gemini(model="models/gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY)
Settings.transformations = [SentenceSplitter(chunk_size=1024)]

# Create index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

# Set up query engine
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    use_async=True,
)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

def rag_query(query: str) -> str:
    """Perform a RAG query and return the response."""
    response = query_engine.query(query)
    return str(response)

def text_generation(query: str) -> str:
    """Generate text using Gemini model."""
    prompt = f"You are a qualified financial consultant, please respond to the following query: {query}"
    response = Settings.llm.complete(prompt)
    return response.text

def hybrid_query(query: str, chat_history: List[Tuple[str, str]]) -> str:
    """Combine RAG and text generation based on the query and chat history."""
    rag_response = rag_query(query)
    
    if "I don't have enough information" in rag_response or len(rag_response) < 50:
        # If RAG doesn't provide a satisfactory answer, use text generation
        return text_generation(query)
    else:
        return rag_response

# Set up Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Hybrid RAG and Text Generation Chatbot for Mutual Fund Advice")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = hybrid_query(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
