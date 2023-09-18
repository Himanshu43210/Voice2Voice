from playsound import playsound

def play_audio_from_id(matched_object_id):
    filename = matched_object_id + ".mp3"
    try:
        playsound(filename)
    except Exception as e:
        print(f"Error playing audio: {e}")

import random
import os
from playsound import playsound
import threading

def play_random_filler(folder_path='./components/fillers'):
    # Given list of fillers
    fillers = ["alright.mp3", "Gotcha.mp3", "hm_hm.mp3", "okay.mp3", "right.mp3", "yeah.mp3", "yes.mp3"]

    # Select a random filler
    chosen_filler = random.choice(fillers)

    # Play the chosen filler using playsound in a separate thread
    threading.Thread(target=playsound, args=(os.path.join(folder_path, chosen_filler),)).start()
