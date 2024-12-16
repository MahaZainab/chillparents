import os
import whisper
import streamlit as st
from gtts import gTTS
from groq import Groq
import tempfile

# Set up Groq API client
client = Groq(
    api_key="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",
)
# Load Whisper model
model = whisper.load_model("base")

def chatbot(audio_file):
    # Transcribe the audio input using Whisper
    transcription = model.transcribe(audio_file)
    user_input = transcription["text"]

    # Generate a response using Llama 8B via Groq API
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
    )
    response_text = chat_completion.choices[0].message.content

    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='ja')
    
    # Create a temporary file to save the response audio
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_audio.name)
    
    return response_text, temp_audio.name

def main():
    st.title("Voice-to-Voice Chatbot")
    st.markdown("Powered by OpenAI Whisper, Llama 8B, and gTTS. Talk to the AI-powered chatbot and get responses in real-time.")

    # Upload audio file
    audio_file = st.file_uploader("Record Your Voice", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        # Display the uploaded audio
        st.audio(audio_file, format="audio/wav")
        
        # Process the audio and generate a response
        response_text, response_audio_file = chatbot(audio_file)
        
        # Display the chatbot response
        st.subheader("Chatbot Response:")
        st.text_area("Response Text", response_text, height=150)
        
        # Play the generated audio response
        st.audio(response_audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
