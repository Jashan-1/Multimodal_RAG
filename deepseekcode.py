import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Union, Tuple
import torch
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, AutoModelForSpeechSeq2Seq
import librosa
import re
from pinecone import Pinecone
from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, LargeBinary, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import fitz  # PyMuPDF
import cv2
from PIL import Image
import numpy as np
import logging
from pymupdf4llm import to_markdown  # Import to_markdown from pymupdf4llm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Database setup
Base = declarative_base()

class ChatBotMedia(Base):
    __tablename__ = 'chatbot_media'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    content_metadata = Column(JSON, nullable=False)  # Store metadata as JSON
    file_content = Column(LargeBinary, nullable=False)  # Store file content as binary
    upload_timestamp = Column(DateTime, nullable=True, default=func.now())

# Database connection
DATABASE_URL = os.getenv("DATABASE_CONNECTION_URL")
print(DATABASE_URL)
from sqlalchemy import create_engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "multimodal"

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document Processor
class DocumentProcessor:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        self.index_name = INDEX_NAME
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(self.index_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.highlight_color = (1, 0.95, 0)  # Yellow highlight color
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        return re.sub(r'\s+', ' ', text)
        
    def process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF and extract text with page numbers and locations using pymupdf4llm"""
        documents = []
        
        # Extract text and metadata using pymupdf4llm
        markdown_text = to_markdown(file_path)
        
        # Split the markdown text into pages (assuming each page is separated by a form feed character)
        pages = markdown_text.split('\f')
        
        for page_num, page_text in enumerate(pages, start=1):
            if page_text.strip():
                documents.append({
                    "text": page_text,
                    "metadata": {
                        "file_path": file_path,
                        "page_num": page_num,
                        "type": "pdf",
                        "timestamp": datetime.now().isoformat()
                    }
                })
        
        # Create embeddings and upsert to Pinecone
        texts = [doc["text"] for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare vectors for Pinecone upsert
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector = {
                "id": f"{file_path}_chunk_{i}_{datetime.now().timestamp()}",  # Unique ID
                "values": embedding,
                "metadata": {**doc["metadata"], "text": doc["text"]}
            }
            vectors.append(vector)
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        
        return documents

    def highlight_pdf(self, pdf_path: str, output_path: str, text_to_highlight: str, page_number: int):
        """Highlight text in a PDF"""
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        # Search for the text and highlight it
        for instance in page.search_for(text_to_highlight):
            page.add_highlight_annot(instance)

        # Save the updated PDF
        doc.save(output_path)
        doc.close()

# Video Processor
class VideoProcessor:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
        
    def process_video(self, video_path: str) -> List[Dict]:
        """Process video and extract transcripts with timestamps"""
        documents = []
        video = VideoFileClip(video_path)
        
        # Process audio in chunks
        chunk_duration = 30  # Process 30 seconds at a time
        total_duration = video.duration
        
        for start_time in range(0, int(total_duration), chunk_duration):
            end_time = min(start_time + chunk_duration, total_duration)
            
            # Extract audio chunk
            subclip = video.subclip(start_time, end_time)
            audio_path = tempfile.mktemp(suffix=".wav")
            subclip.audio.write_audiofile(audio_path)
            
            # Process audio chunk
            audio_input, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_features, return_timestamps=True)
            
            # Decode the outputs
            decoded_segments = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                output_offsets=True
            )
            print(decoded_segments,type(decoded_segments))
            # Adjust timestamps relative to the full video
            for segment in decoded_segments:
            
                if isinstance(segment, dict) and "text" in segment and "offsets" in segment:
                    print("in if")
                    if "start" in segment["offsets"] and "end" in segment["offsets"]:
                        # Extract start and end times from the offsets
                        segment_start = segment["offsets"]["start"]
                        print(segment_start)
                        segment_end = segment["offsets"]["end"]
                    else: 
                        print("in else")
                        segment_start= segment["offsets"]["timestamp"][0]
                        segment_end= segment["offsets"]["timestamp"][1]
                        print(segment_start)
                        
                    # Adjust timestamps relative to the full video
                    adjusted_start = start_time + segment_start
                    adjusted_end = start_time + segment_end
                    
                    documents.append({
                        "text": segment["text"],
                        "metadata": {
                            "file_path": video_path,
                            "start_time": adjusted_start,
                            "end_time": adjusted_end,
                            "type": "video",
                            "timestamp": datetime.now().isoformat()
                        }
                    })
            
            os.remove(audio_path)
        
        video.close()
        return documents

    def create_video_highlight(self, video_path: str, segments: List[Dict]) -> str:
        """Create a compilation of relevant video segments"""
        clips = []
        video = VideoFileClip(video_path)

        # print(segments)
        for segment in segments:
            start_time = segment['metadata']['start_time']
            end_time = segment['metadata']['end_time']
            clip = video.subclip(start_time, end_time)
            
            # Add text overlay with timestamp
            clip = clip.set_position('bottom').set_duration(end_time - start_time)
            clips.append(clip)
        
        # Concatenate all clips
        if clips:
            final_clip = concatenate_videoclips(clips)
            output_path = tempfile.mktemp(suffix=".mp4")
            final_clip.write_videofile(output_path)
            
            # Clean up
            video.close()
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return output_path
        return None

# Training System
class TrainingSystem:
    def __init__(self, openai_api_key: str):
        self.doc_processor = DocumentProcessor(openai_api_key=openai_api_key)
        self.video_processor = VideoProcessor()
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-4"
        )

    def upload_document(self, file_path: str, file_type: str, user_id: str):
        """Upload and process a document or video."""
        try:
            # Read file content
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Process file based on type
            if file_type == "pdf":
                documents = self.doc_processor.process_pdf(file_path)
            elif file_type in ["mp4", "mov", "avi"]:
                documents = self.video_processor.process_video(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Store metadata in database
            metadata = {
                "documents": documents,
                "total_segments": len(documents)
            }

            # Insert into PostgreSQL
            session = SessionLocal()
            new_media = ChatBotMedia(
                file_path=file_path,
                file_type=file_type,
                user_id=user_id,
                content_metadata=json.dumps(metadata),  # Store metadata as JSON string
                file_content=file_content  # Store file content as binary
            )
            session.add(new_media)
            session.commit()
            session.close()

            return True

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")  # Convert exception to string
            return False

    def query_documents(self, query: str, user_id: str) -> Dict:
        """Query documents and generate a response using GPT-4."""
        # Generate query embedding
        query_embedding = self.doc_processor.embeddings.embed_query(query)
        logger.info(f"Query embedding generated: {query_embedding[:5]}...")  # Log embedding

        # Query Pinecone for relevant chunks
        results = self.doc_processor.index.query(
            vector=query_embedding,
            top_k=5,  # Retrieve top 5 relevant chunks
            include_metadata=True
        )

        logger.info(f"Pinecone query results: {results}")  # Log Pinecone results

        # Check if results are empty
        if not results['matches']:
            logger.warning("No matches found in Pinecone for the query.")
            return {
                "answer": "I couldn't find any relevant information in the documents.",
                "pdf_sources": [],
                "video_sources": []
            }

        # Group results by document type and file
        pdf_results = {}
        video_results = {}

        for match in results['matches']:
            metadata = match['metadata']
            if metadata['type'] == 'pdf':
                file_path = metadata['file_path']
                if file_path not in pdf_results:
                    pdf_results[file_path] = []
                pdf_results[file_path].append(match)
            else:  # video
                file_path = metadata['file_path']
                if file_path not in video_results:
                    video_results[file_path] = []
                video_results[file_path].append(match)

        logger.info(f"PDF results: {pdf_results}")
        logger.info(f"Video results: {video_results}")

        # Generate response using metadata from Pinecone
        response = {
            "answer": self.generate_answer(query, results),
            "pdf_sources": [],
            "video_sources": []
        }

        # Process PDF results
        for file_path, matches in pdf_results.items():
            # Fetch file content from PostgreSQL
            session = SessionLocal()
            file_record = session.query(ChatBotMedia).filter(ChatBotMedia.file_path == file_path).first()
            session.close()

            if file_record:
                # Save file content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_record.file_content)
                    tmp_file_path = tmp_file.name

                # Highlight relevant text in the PDF
                highlighted_pdf_path = tempfile.mktemp(suffix=".pdf")
                for match in matches:
                    page_num = match['metadata']['page_num']
                    text_to_highlight = match['metadata']['text']
                    self.doc_processor.highlight_pdf(tmp_file_path, highlighted_pdf_path, text_to_highlight, page_num)

                # Crop and display relevant pages
                for match in matches:
                    page_num = match['metadata']['page_num']
                    doc = fitz.open(highlighted_pdf_path)
                    page = doc.load_page(page_num - 1)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes()

                    response["pdf_sources"].append({
                        "file_path": file_path,
                        "page_num": page_num,
                        "highlighted_page": img_data,
                        "matches": [match['metadata']]
                    })

                os.unlink(tmp_file_path)  # Clean up temporary file
                os.unlink(highlighted_pdf_path)  # Clean up highlighted PDF

        # Process video results
        for file_path, matches in video_results.items():
            # Fetch file content from PostgreSQL
            session = SessionLocal()
            file_record = session.query(ChatBotMedia).filter(ChatBotMedia.file_path == file_path).first()
            session.close()

            if file_record:
                # Save file content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(file_record.file_content)
                    tmp_file_path = tmp_file.name

                # Crop and display relevant segments
                video = VideoFileClip(tmp_file_path)
                clips = []
                for match in matches:
                    start_time = match['metadata']['start_time']
                    end_time = match['metadata']['end_time']
                    clip = video.subclip(start_time, end_time)
                    clips.append(clip)

                if clips:
                    final_clip = concatenate_videoclips(clips)
                    output_path = tempfile.mktemp(suffix=".mp4")
                    final_clip.write_videofile(output_path)

                    response["video_sources"].append({
                        "file_path": file_path,
                        "highlight_path": output_path,
                        "segments": [match['metadata'] for match in matches]
                    })

                os.unlink(tmp_file_path)  # Clean up temporary file

        return response

    def generate_answer(self, query: str, results: Dict) -> str:
        """Generate an answer using GPT-4 based on retrieved documents"""
        # Combine all retrieved text chunks into a single context
        context = "\n\n".join([match['metadata'].get('text', '') for match in results['matches']])

        # Log the context for debugging
        logger.info(f"Context for query '{query}':\n{context}")

        # If context is empty, return a default response
        if not context.strip():
            return "I couldn't find any relevant information in the documents to answer your question."

        # Create a more structured prompt
        prompt = f"""You are an expert assistant. Based on the following context, answer the question in detail. If the context does not provide enough information, say "I don't have enough information to answer that question."

        Context:
        {context}

        Question:
        {query}

        Answer:"""

        # Log the prompt for debugging
        logger.info(f"Prompt for query '{query}':\n{prompt}")

        # Generate the response using GPT-4
        response = self.llm.predict(prompt)
        return response

# Page Rendering Functions
def render_upload_page(training_system):
    st.header("Upload Training Materials")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or Video files",
        type=["pdf", "mp4", "mov", "avi"],
        accept_multiple_files=True  # Allow multiple files
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
                
                file_info = {
                    "name": uploaded_file.name,
                    "path": file_path,
                    "type": uploaded_file.type.split("/")[1],
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            if training_system.upload_document(
                file_path=file_path,
                file_type=file_info["type"],
                user_id=st.session_state.get("user_id", "default_user")
            ):
                st.success(f"File '{file_info['name']}' uploaded and processed successfully!")
                if file_info not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file_info)
            else:
                st.error(f"Error processing file '{file_info['name']}'")
                os.unlink(file_path)

def render_view_materials_page():
    st.header("View Training Materials")
    
    if not st.session_state.uploaded_files:
        st.info("No training materials uploaded yet. Please upload some files first.")
        return
    
    df_data = [{
        "File Name": file["name"],
        "Type": file["type"].upper(),
        "Upload Time": file["upload_time"]
    } for file in st.session_state.uploaded_files]
    
    st.table(df_data)
    
    if st.session_state.uploaded_files:
        selected_file = st.selectbox(
            "Select a file to view",
            options=[file["name"] for file in st.session_state.uploaded_files]
        )
        
        file_info = next(
            file for file in st.session_state.uploaded_files 
            if file["name"] == selected_file
        )
        
        if file_info["type"] == "pdf":
            try:
                doc = fitz.open(file_info["path"])
                page_count = len(doc)
                if page_count > 1:
                    page_num = st.slider("Select page", 1, page_count, 1)
                    display_pdf(file_info["path"], page_num)
                elif page_count == 1:
                    st.write("Displaying the only page in the PDF:")
                    display_pdf(file_info["path"], 1)
                else:
                    st.warning("The selected PDF has no pages.")
            except Exception as e:
                st.error(f"Error displaying PDF: {str(e)}")  # Convert exception to string
        else:  # video
            try:
                st.video(file_info["path"])
            except Exception as e:
                st.error(f"Error playing video: {str(e)}")  # Convert exception to string

def render_chatbot_page(training_system):
    st.header("Chat with Your Documents")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display media if present
            if "media" in message:
                for pdf_source in message["media"].get("pdf_sources", []):
                    with st.expander(f"PDF Source: {os.path.basename(pdf_source['file_path'])} - Page {pdf_source['page_num']}"):
                        st.image(pdf_source["highlighted_page"], caption=f"Page {pdf_source['page_num']}", use_column_width=True)
                        
                for video_source in message["media"].get("video_sources", []):
                    with st.expander(f"Video Source: {os.path.basename(video_source['file_path'])}"):
                        st.video(video_source["highlight_path"])
                        st.write("Relevant Segments:")
                        for segment in video_source["segments"]:
                            st.write(f"- {segment['text']} ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response with media
        response = training_system.query_documents(
            query=prompt,
            user_id=st.session_state.get("user_id", "default_user")
        )
        
        # Format response text
        response_text = response["answer"]
        
        # Add response with media to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "media": {
                "pdf_sources": response["pdf_sources"],
                "video_sources": response["video_sources"]
            }
        })
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
            # Display PDF sources
            for pdf_source in response["pdf_sources"]:
                with st.expander(f"PDF Source: {os.path.basename(pdf_source['file_path'])} - Page {pdf_source['page_num']}"):
                    st.image(pdf_source["highlighted_page"], caption=f"Page {pdf_source['page_num']}", use_column_width=True)
                    
            # Display video sources
            for video_source in response["video_sources"]:
                with st.expander(f"Video Source: {os.path.basename(video_source['file_path'])}"):
                    st.video(video_source["highlight_path"])
                    st.write("Relevant Segments:")
                    for segment in video_source["segments"]:
                        st.write(f"- {segment['text']} ({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")

def display_pdf(file_path: str, page_num: int):
    """Display a specific page of a PDF"""
    doc = fitz.open(file_path)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap()
    st.image(pix.tobytes(), use_column_width=True)

def create_streamlit_app():
    st.set_page_config(page_title="Training Document System", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Upload Materials", "View Materials", "Chat with Documents"]
    )
    
    # Initialize system
    training_system = TrainingSystem(openai_api_key=OPENAI_API_KEY)
    
    # Main content based on selected page
    if page == "Upload Materials":
        render_upload_page(training_system)
    elif page == "View Materials":
        render_view_materials_page()
    else:  # Chat with Documents
        render_chatbot_page(training_system)
    
    # Display current user
    st.sidebar.divider()
    st.sidebar.text(f"Current User: {st.session_state.get('user_id', 'default_user')}")

if __name__ == "__main__":
    create_streamlit_app()