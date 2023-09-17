from playsound import playsound

def play_audio_from_id(matched_object_id):
    filename = matched_object_id + ".mp3"
    try:
        playsound(filename)
    except Exception as e:
        print(f"Error playing audio: {e}")

