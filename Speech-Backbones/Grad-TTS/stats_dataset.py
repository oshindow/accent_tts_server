import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

def get_audio_lengths(folder_path):
    audio_lengths = []

    # Traverse all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an audio file
            if file.endswith(('.mp3', '.wav', '.flac', '.aac')):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_file(file_path)
                length_in_seconds = len(audio) / 1000  # Length in seconds
                audio_lengths.append(length_in_seconds)

    return audio_lengths

def plot_audio_lengths(audio_lengths):
    plt.hist(audio_lengths, bins=30, edgecolor='black')
    plt.title('Distribution of Audio Lengths')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Number of Files')
    plt.savefig('LJSpeech.png')

# Specify the folder path
folder_path = '/data2/xintong/tts/LJSpeech-1.1/wavs'

# Get audio lengths
audio_lengths = get_audio_lengths(folder_path)
print(len(audio_lengths))
# Plot the lengths
plot_audio_lengths(audio_lengths)


