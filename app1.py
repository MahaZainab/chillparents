import whisper
import streamlit as st
from gtts import gTTS
from groq import Groq
import tempfile
import numpy as np
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioFrame

# Set up Groq API client
client = Groq(
    api_key="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",
)
# Load Whisper model
model = whisper.load_model("base")

# Function to handle chatbot interaction
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

# Handle incoming audio recording via WebRTC
def audio_callback(frame: AudioFrame):
    if frame is None:
        return None
    
    # Convert the audio frame to a numpy array (audio data)
    audio_data = frame.to_ndarray()

    # Save the audio to a temporary file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_data.tofile(temp_audio_file.name)  # Save the audio data to the file

    # Process the audio and get a response
    response_text, response_audio_file = chatbot(temp_audio_file.name)

    # Display the chatbot response
    st.subheader("Chatbot Response:")
    st.text_area("Response Text", response_text, height=150)
    st.audio(response_audio_file, format="audio/mp3")

# Streamlit UI
def main():
    st.title("Voice-to-Voice Chatbot")
    st.markdown("Powered by OpenAI Whisper, Llama 8B, and gTTS. Talk to the AI-powered chatbot and get responses in real-time.")

    # Set up WebRTC for audio recording
    webrtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    st.subheader("Record your voice")

    # Set up WebRTC streamer with voice recording
    webrtc_streamer(
        key="audio-chatbot",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=webrtc_config,
        audio_receiver_size=1024,
        audio_callback=audio_callback  # Use the correct callback here
    )

if __name__ == "__main__":
    main()
