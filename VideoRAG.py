import streamlit as st
import os
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Extract frames from video
def extract_frames(video_path, interval=1):
    frames = []
    with VideoFileClip(video_path) as video:
        duration = video.duration
        for i in range(0, int(duration), interval):
            frame = video.get_frame(i)
            frames.append((i, Image.fromarray(frame)))
    return frames

# Transcribe audio from video using OpenAI's Whisper API
def transcribe_audio(video_path):
    # Extract audio from video
    with VideoFileClip(video_path) as video:
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
        
        # Transcribe the audio using OpenAI's Whisper API
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        
        # Clean up the temporary audio file
        os.remove(audio_path)
        
        return transcript["text"]

# Updated split_transcript function to include timestamps for each segment
def split_transcript(transcript, words_per_segment=50):
    words = transcript.split()
    segments = []
    start_time = 0  # Track the start timestamp for each segment
    words_per_second = 2.5  # Average speaking rate (words per second)
    segment_duration = words_per_segment / words_per_second
    
    for i in range(0, len(words), words_per_segment):
        segment_text = " ".join(words[i:i + words_per_segment])
        end_time = start_time + segment_duration
        segments.append((start_time, end_time, segment_text))  # (start, end, text)
        start_time = end_time  # Update for the next segment

    return segments

# Get embeddings for text using OpenAI
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Find the most relevant segment using semantic search
def find_relevant_segment(query, segments):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    segment_embeddings = np.array([get_embedding(segment[2]) for segment in segments])
    similarities = cosine_similarity(query_embedding, segment_embeddings).flatten()
    most_relevant_index = np.argmax(similarities)
    return segments[most_relevant_index], similarities[most_relevant_index]

# Generate response using OpenAI's GPT-3.5 Turbo
def generate_response(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\n{query}"},
        ],
        max_tokens=150,
    )
    return response.choices[0].message["content"].strip()

# Updated subclip extraction logic
def generate_subclip(video_path, start_time, end_time, margin=3):
    with VideoFileClip(video_path) as video:
        duration = video.duration
        start_time = max(start_time - margin, 0)  # Add margin before the relevant start time
        end_time = min(end_time + margin, duration)  # Add margin after the relevant end time
        
        # Extract subclip and save
        subclip = video.subclip(start_time, end_time)
        subclip_path = f"subclip_{int(start_time)}-{int(end_time)}.mp4"
        subclip.write_videofile(subclip_path, audio_codec="aac")
        return subclip_path

# Streamlit app
def main():
    st.title("Multimodal RAG: Chat with Videos")
    video_file = st.file_uploader("Upload a video", type=["mp4"])

    # Initialize session state to store conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if video_file is not None:
        video_path = f"./{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        st.video(video_path)
        
        # Transcribe audio using OpenAI's Whisper API
        if "transcript" not in st.session_state:
            st.write("Transcribing audio... (this may take a moment)")
            st.session_state.transcript = transcribe_audio(video_path)
        st.write("Transcript:", st.session_state.transcript)

        # Split the transcript into timestamped segments
        if "segments" not in st.session_state:
            st.session_state.segments = split_transcript(st.session_state.transcript)

        # Handle user queries
        query = st.text_input("Enter your query about the video:")
        if query:
            # Add the query to the conversation history
            st.session_state.conversation.append(("user", query))

            # Find the most relevant segment
            relevant_segment, similarity = find_relevant_segment(query, st.session_state.segments)
            start_time, end_time, relevant_text = relevant_segment
            st.write(f"Relevant segment: {relevant_text}")
            st.write(f"Relevant timestamp: {start_time:.2f} - {end_time:.2f} seconds")

            # Generate response using GPT-3.5
            context = f"Transcript: {st.session_state.transcript}\nRelevant text: {relevant_text}"
            response = generate_response(query, context)
            st.session_state.conversation.append(("assistant", response))

            # Display conversation history
            st.write("### Conversation History")
            for role, message in st.session_state.conversation:
                if role == "user":
                    st.write(f"**You:** {message}")
                else:
                    st.write(f"**Assistant:** {message}")

            # Generate and display the subclip
            subclip_path = generate_subclip(video_path, start_time, end_time)
            st.video(subclip_path)

if __name__ == "__main__":
    main()