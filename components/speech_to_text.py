# Import the necessary libraries
from google.cloud import speech
import os
import sounddevice as sd
from dotenv import load_dotenv

# Load environment variables from .env file for secure access
load_dotenv()

def transcribe_stream():
    """Transcribe real-time audio stream using Google Cloud Speech API."""
    
    # Retrieve path for Google Cloud credentials from environment variable
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # If the credentials path isn't set in the .env file, raise an error
    if not creds_path:
        raise ValueError("Credentials path is not set in .env file.")
    
    # Initialize Google Cloud Speech client
    client = speech.SpeechClient()
    
    # Configuration settings for the speech recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Encoding format of audio data
        sample_rate_hertz=44100,                                 # Sample rate of the audio
        language_code="en-US",                                   # Language for transcription
        use_enhanced=True                                        # Enable enhanced model for faster recognition
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)  # Include interim results

    def response_generator():
        """Generator function to continuously yield audio data for the streaming request."""
        
        # Capture audio input with specified configurations
        with sd.InputStream(samplerate=44100, channels=1, device=2, dtype="int16") as stream:
            while True:
                data, _ = stream.read(512)                       # Read audio data from stream
                yield speech.StreamingRecognizeRequest(audio_content=data.tobytes())  # Yield audio data for recognition

    # Initiate streaming recognition with the generated audio data
    responses = client.streaming_recognize(streaming_config, response_generator())

    # Process the responses to fetch the transcriptions
    for response in responses:
        if response.results:                                     # If there's any result in the response
            result = response.results[0]
            if result.is_final:                                 # Check if the result is the final transcription
                transcription_text = result.alternatives[0].transcript
                print('Transcription: {}'.format(transcription_text))
                return transcription_text

# Begin the transcription process
print("start speaking...")
prompt_text = transcribe_stream()
print(prompt_text)
