import os
import tempfile
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st
from pymupdf4llm import to_markdown
from langchain.text_splitter import MarkdownTextSplitter
import fitz  # PyMuPDF for embedding

# Set page config as the first Streamlit command
st.set_page_config(page_title="Multi-PDF RAG Bot", layout="wide")

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multimodal"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Please set OPENAI_API_KEY and PINECONE_API_KEY in your environment variables.")
    st.stop()

openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
pc = Pinecone(api_key=PINECONE_API_KEY)  # Initialize Pinecone client

# Check if Pinecone index exists and create it if not
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)  # Connect to the Pinecone index

# Initialize Streamlit session state for uploaded PDFs
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

# Markdown splitter
splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to extract text and chunk it
def extract_and_chunk_pdf(file_path: str) -> List[Dict]:
    """Extract text from a PDF, convert it to Markdown, and split into chunks."""
    try:
        # Convert the PDF to Markdown
        markdown_text = to_markdown(file_path)
        
        # Chunk the Markdown text
        chunks = splitter.create_documents([markdown_text])
        
        # Create documents with metadata for each chunk
        documents = [
            {
                "text": chunk.page_content.strip(),
                "metadata": {
                    "file_path": file_path,
                    "chunk_id": idx,
                }
            }
            for idx, chunk in enumerate(chunks)
        ]
        
        print(f"Extracted and chunked {len(documents)} chunks from PDF: {file_path}")  # Debugging
        return documents
    except Exception as e:
        st.error(f"Error extracting and chunking PDF: {e}")
        return []

# Function to generate embeddings
def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings for a given text using OpenAI."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Function to process and ingest PDF content into Pinecone
def upload_and_process_pdf(file_path: str, file_name: str):
    """Extract text, generate embeddings, and store in Pinecone."""
    try:
        documents = extract_and_chunk_pdf(file_path)  # Extract and chunk the PDF
        
        # Generate embeddings and prepare vectors
        vectors = []
        for doc in documents:
            embedding = generate_embeddings(doc["text"])  # Generate embedding for chunk text
            if embedding:
                vectors.append({
                    "id": f"{file_name}_chunk_{doc['metadata']['chunk_id']}",
                    "values": embedding,
                    "metadata": {
                        "text": doc["text"],  # Ensure the 'text' key is included in metadata
                        "file_path": file_path,
                        "chunk_id": doc['metadata']['chunk_id'],
                        "file_name": file_name
                    }
                })
        
        # Upsert vectors to Pinecone
        if vectors:
            index.upsert(vectors=vectors)
            st.success(f"Ingested {len(vectors)} chunks from {file_name} into Pinecone!")
        
        # Track uploaded PDFs
        st.session_state.uploaded_pdfs.append({
            "file_path": file_path,
            "file_name": file_name,
            "num_chunks": len(documents)
        })
        return True
    except Exception as e:
        st.error(f"Error processing PDF {file_name}: {e}")
        return False

# Streamlit app main function
def main():
    st.title("Multi-PDF RAG Bot")
    
    # Sidebar for uploading PDFs
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()  # Ensure all data is written to the file
                if upload_and_process_pdf(tmp_file.name, uploaded_file.name):
                    st.sidebar.success(f"Uploaded and processed {uploaded_file.name} successfully!")
    
    # Sidebar for displaying uploaded PDFs
    st.sidebar.header("Uploaded PDFs")
    if st.session_state.uploaded_pdfs:
        for pdf in st.session_state.uploaded_pdfs:
            st.sidebar.write(f"{pdf['file_name']} ({pdf['num_chunks']} chunks)")
    else:
        st.sidebar.info("No PDFs uploaded yet.")
    
    # Query interface
    st.header("Query Your PDFs")
    query = st.text_input("Enter your question:")
    if query:
        # Generate embeddings for the query
        query_embedding = generate_embeddings(query)
        
        # Query Pinecone for relevant chunks
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if results.matches:
            # Extract context from the retrieved matches and deduplicate
            context_set = set()
            for match in results.matches:
                text = match.metadata.get("text", "").strip()
                if text:
                    context_set.add(text)
            
            context = "\n\n".join(context_set)
            
            if not context:
                st.warning("No valid context found for the query.")
                return
            
            # Prepare the prompt for GPT-4
            prompt = f"""
            You are an assistant for question-answering tasks after analyzing PDF(s). 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Question: {query} 
            Context: {context} 
            Answer:
            """
            
            # Call GPT-4 to generate the answer
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # Use "gpt-4-turbo" or "gpt-4" depending on availability
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,  # Limit the response length
                    temperature=0.7  # Control creativity
                )
                
                # Extract the answer from the response
                answer = response['choices'][0]['message']['content'].strip()
                
                # Display the answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display context in an expandable section
                with st.expander("Show context"):
                    st.write(context)
                
            except Exception as e:
                st.error(f"Error generating answer with GPT-4: {e}")
        
        else:
            st.warning("No results found.")

if __name__ == "__main__":
    main()
