import os
import tempfile
from typing import List, Dict
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import fitz  # PyMuPDF
from datetime import datetime

# Set page config as the first Streamlit command
st.set_page_config(page_title="Multi-PDF RAG Bot", layout="wide")

# Load environment variables
load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multimodal"

# Validate environment variables
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Please set OPENAI_API_KEY and PINECONE_API_KEY in your environment variables.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if Pinecone index exists and create it if not
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Get Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize session state
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = {}

def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings using OpenAI's text-embedding-ada-002 model."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

class PDFProcessor:
    def process_document(self, file_path: str, file_name: str) -> List[Dict]:
        """Process PDF and extract text with metadata."""
        documents = []
        with fitz.open(file_path) as pdf_doc:
            # Extract metadata
            metadata = pdf_doc.metadata

            # Extract table of contents
            toc = pdf_doc.get_toc()

            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                
                # Get text blocks with positions
                blocks = page.get_text("dict")["blocks"]
                
                for block_num, block in enumerate(blocks):
                    if "lines" in block:
                        # Extract text and its position
                        text = " ".join([
                            span["text"] for line in block["lines"] 
                            for span in line.get("spans", [])
                        ])
                        
                        if text.strip():
                            documents.append({
                                "text": text,
                                "metadata": {
                                    "file_path": file_path,
                                    "file_name": file_name,
                                    "page_num": page_num + 1,
                                    "block_num": block_num,
                                    "bbox": block["bbox"],
                                    "type": "pdf",
                                    "title": metadata.get("title", ""),
                                    "author": metadata.get("author", "")
                                }
                            })
            
            # Store metadata in session state
            st.session_state.pdf_metadata[file_name] = {
                "metadata": metadata,
                "toc": toc,
                "num_pages": len(pdf_doc)
            }
            
        return documents

    def highlight_text_on_page(self, file_path: str, page_num: int, text_to_highlight: str) -> bytes:
        """Create highlighted version of PDF page."""
        try:
            doc = fitz.open(file_path)
            page = doc[page_num - 1]
            
            # Search for text on page
            text_instances = page.search_for(text_to_highlight)
            
            # Add yellow highlights
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=(1, 1, 0))  # Yellow color
                highlight.update()
            
            # Render page with highlights
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_bytes = pix.tobytes()
            doc.close()
            return img_bytes
            
        except Exception as e:
            st.error(f"Error highlighting text on page: {e}")
            return None

def process_and_store_pdf(file_path: str, file_name: str) -> bool:
    """Process PDF and store in Pinecone."""
    try:
        # Process the document
        documents = pdf_processor.process_document(file_path, file_name)
        
        # Generate embeddings and store
        vectors = []
        for doc in documents:
            embedding = generate_embeddings(doc["text"])
            if embedding:
                vector = {
                    "id": f"{file_name}_p{doc['metadata']['page_num']}_b{doc['metadata']['block_num']}",
                    "values": embedding,
                    "metadata": {**doc["metadata"], "text": doc["text"]}
                }
                vectors.append(vector)
        
        # Upsert to Pinecone in batches
        if vectors:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
            
            st.success(f"Successfully processed and stored {len(vectors)} text blocks from {file_name}")
            return True
        
        return False
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def query_documents(query: str) -> Dict:
    """Query documents and generate response."""
    try:
        # Generate query embedding
        query_embedding = generate_embeddings(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if not results['matches']:
            return {
                "answer": "No relevant information found.",
                "context": "",
                "highlights": []
            }
        
        # Extract context and prepare highlights
        context = []
        highlights = []
        
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.get('text', '')
            context.append(f"[Score: {match['score']:.2f}] {text}")
            
            highlights.append({
                "file_path": metadata['file_path'],
                "file_name": metadata['file_name'],
                "page_num": metadata['page_num'],
                "text": text
            })
        
        # Generate answer using GPT-4
        prompt = f"""Based on the following context, answer the question concisely.
        If you don't know, say 'I don't have enough information.'

        Context: {' '.join(context)}
        
        Question: {query}
        
        Answer:"""
        
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        answer = completion.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "context": "\n\n".join(context),
            "highlights": highlights
        }
        
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return {"answer": str(e), "context": "", "highlights": []}

# Initialize PDF processor
class PDFProcessor:
    def process_document(self, file_path: str, file_name: str):
        """Process PDF and extract text with metadata."""
        documents = []
        with fitz.open(file_path) as pdf_doc:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                text = page.get_text()
                documents.append({
                    "text": text,
                    "metadata": {
                        "file_path": file_path,
                        "file_name": file_name,
                        "page_num": page_num + 1,  # Pages are 1-indexed
                        "block_num": 1  # Placeholder for block number
                    }
                })
        return documents

    def highlight_text_on_page(self, file_path: str, page_num: int, text_to_highlight: str):
        """Highlight the relevant text on the specified PDF page and return the image."""
        try:
            pdf_doc = fitz.open(file_path)
            page = pdf_doc.load_page(page_num - 1)  # Pages are 0-indexed in PyMuPDF

            # Search for the text on the page
            text_instances = page.search_for(text_to_highlight)

            # Highlight each instance of the text
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)

            # Render the highlighted page as an image
            pix = page.get_pixmap()
            image_path = tempfile.mktemp(suffix=".png")
            pix.save(image_path)
            return image_path
        except Exception as e:
            st.error(f"Error highlighting text on page: {e}")
            return None

    @property
    def metadata_cache(self):
        """Placeholder for metadata cache logic."""
        return {}

pdf_processor = PDFProcessor()

def generate_embeddings(text: str):
    """Generate embeddings for the given text."""
    # Placeholder for embedding generation logic
    return [0.0] * 1536  # Dummy embedding

def process_and_store_pdf(file_path: str, file_name: str) -> bool:
    """Process PDF and store its information."""
    try:
        # Process the document with enhanced metadata
        documents = pdf_processor.process_document(file_path, file_name)

        # Generate embeddings and store in vector database
        texts = [doc["text"] for doc in documents]
        embeddings = [generate_embeddings(text) for text in texts]

        # Prepare vectors for storage
        vectors = []
        for doc, emb in zip(documents, embeddings):
            vector = {
                "id": f"{file_name}_p{doc['metadata']['page_num']}_b{doc['metadata']['block_num']}",
                "values": emb,
                "metadata": doc["metadata"]
            }
            vectors.append(vector)

        # Store in vector database (placeholder)
        # index.upsert(vectors=vectors)

        # Store in session state
        if "pdf_metadata" not in st.session_state:
            st.session_state.pdf_metadata = {}
        st.session_state.pdf_metadata[file_name] = {"num_pages": len(documents), "toc": []}

        return True
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def query_documents(query: str) -> Dict:
    """Enhanced query processing with page-specific results."""
    try:
        # Generate query embedding
        query_embedding = generate_embeddings(query)

        # Query vector database (placeholder)
        results = {
            "matches": [
                {
                    "metadata": {
                        "file_path": "path/to/sample.pdf",
                        "file_name": "sample.pdf",
                        "page_num": 1,
                        "text": "This is a sample text from the PDF."
                    }
                }
            ]
        }

        if not results['matches']:
            return {
                "answer": "No relevant information found.",
                "context": "",
                "highlights": []
            }

        # Extract context and prepare highlights
        contexts = []
        highlights = []

        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.get('text', '')
            contexts.append(text)

            # Prepare highlight information
            highlights.append({
                "file_path": metadata['file_path'],
                "file_name": metadata['file_name'],
                "page_num": metadata['page_num'],
                "text": text
            })

        # Combine context
        context = "\n\n".join(contexts)

        # Generate answer
        prompt = f"""Based on the following context, answer the question concisely.
        If you don't know, say 'I don't have enough information.'

        Context: {context}
        Question: {query}
        Answer:"""

        response = client.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "context": context,
            "highlights": highlights
        }
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return {"answer": str(e), "context": "", "highlights": []}

def main():
    st.title("Enhanced PDF ChatBot")

    # File upload section
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                if process_and_store_pdf(tmp_file.name, uploaded_file.name):
                    # Display PDF metadata
                    st.sidebar.write(f"**{uploaded_file.name}**")
                    metadata = st.session_state.pdf_metadata[uploaded_file.name]
                    st.sidebar.write(f"Pages: {metadata['num_pages']}")
                    if metadata['toc']:
                        with st.sidebar.expander("Table of Contents"):
                            for entry in metadata['toc']:
                                st.write(f"{'  ' * entry[0]}{entry[1]} (p.{entry[2]})")

    # Create two columns for chat and PDF display
    chat_col, pdf_col = st.columns([1, 1])

    with chat_col:
        st.header("Ask Questions")
        query = st.text_input("Enter your question:")

        if query:
            response = query_documents(query)
            st.write("**Answer:**", response["answer"])

            with st.expander("Show Context"):
                st.write(response["context"])

    with pdf_col:
        if query and response.get("highlights"):
            st.header("Relevant Pages")
            for highlight in response["highlights"]:
                with st.expander(f"Page {highlight['page_num']} from {highlight['file_name']}"):
                    # Display highlighted page
                    highlighted_page = pdf_processor.highlight_text_on_page(
                        highlight['file_path'],
                        highlight['page_num'],
                        highlight['text']
                    )
                    if highlighted_page:
                        st.image(highlighted_page)

if __name__ == "__main__":
    main()