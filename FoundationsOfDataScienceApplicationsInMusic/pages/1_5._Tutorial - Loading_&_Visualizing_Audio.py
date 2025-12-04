import streamlit as st
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'images')

st.title("Tutorial: Loading & Visualizing Audio with librosa")

st.markdown("""
This tutorial introduces you to **librosa**, a powerful Python library for audio analysis.
You'll learn how to load audio files and create visualizations that reveal the structure
of sound—skills fundamental to computational music analysis.

This tutorial is designed for music students with little to no programming experience.
Run the code in Google Colab for free (no local setup required).
""")

st.header("About librosa")
st.markdown("""
**librosa** is an open-source Python library designed for music and audio analysis. Developed
by researchers at Columbia University, it provides tools that bridge the gap between audio
signal processing and music information retrieval (MIR).

### Key Capabilities:
- **Audio Loading**: Read audio files in various formats (WAV, MP3, OGG, etc.) and convert them to numerical arrays
- **Visualization**: Create waveforms, spectrograms, and other visual representations of audio
- **Feature Extraction**: Extract musical features like tempo, pitch, and timbre
- **Time-Frequency Analysis**: Perform spectral analysis using Short-Time Fourier Transform (STFT)
- **Integration**: Works seamlessly with NumPy, SciPy, and matplotlib for scientific computing and visualization

For music students, librosa opens doors to understanding audio from a computational perspective,
enabling analysis that would be impossible by ear alone. Check the [official documentation](https://librosa.org/doc/latest/index.html) for more.
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
In Google Colab, install librosa and supporting libraries. This step may take a minute:
""")
st.code("""
!pip install librosa
!pip install soundfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
print("Libraries loaded successfully!")
""", language="python")

st.header("Step 2: Load an Audio File")
st.markdown("""
librosa makes loading audio simple. The `librosa.load()` function returns two things:
- **y**: The audio signal as a NumPy array (a sequence of numbers representing the waveform)
- **sr**: The sample rate (how many samples per second)

You can use librosa's built-in example files or upload your own. Let's start with a built-in example:
""")
st.code("""
# Load a built-in example (trumpet sound)
y, sr = librosa.load(librosa.ex('trumpet'))

# Print basic information about the audio
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(y) / sr:.2f} seconds")
print(f"Number of samples: {len(y)}")

# Listen to the audio
Audio(y, rate=sr)
""", language="python")

st.markdown("""
**Understanding Sample Rate:**
The sample rate tells us how many "snapshots" of the sound wave were taken per second.
CD quality audio uses 44,100 samples per second (44.1 kHz). librosa defaults to 22,050 Hz
to reduce file size while maintaining good quality for analysis.
""")

st.header("Step 3: Visualize the Waveform")
st.markdown("""
A **waveform** shows amplitude (loudness) over time. It's the most basic visualization
of audio and resembles what you might see in audio editing software.
""")
st.code("""
# Create a figure with a specific size
plt.figure(figsize=(14, 4))

# Plot the waveform using librosa's display function
librosa.display.waveshow(y, sr=sr, alpha=0.8)

# Add labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.tight_layout()
plt.show()
""", language="python")

# Show example output image
waveform_image = os.path.join(IMAGES_DIR, 'tutorial_5_waveform.png')
if os.path.exists(waveform_image):
    st.image(waveform_image, caption="Example output: Audio waveform showing amplitude over time", use_container_width=True)

st.markdown("""
**What to look for in a waveform:**
- **Peaks**: Loud moments in the music
- **Quiet sections**: Near-zero amplitude areas indicate silence or soft passages
- **Attack and decay**: The shape of individual notes reveals articulation style
""")

st.header("Step 4: Create a Spectrogram")
st.markdown("""
A **spectrogram** shows frequency content over time. It reveals which pitches are present
at each moment—information invisible in a simple waveform.

The x-axis is time, the y-axis is frequency (pitch), and the color indicates intensity (loudness).
""")
st.code("""
# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert amplitude to decibels for better visualization
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Create the spectrogram plot
plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
""", language="python")

# Show example output image
spectrogram_image = os.path.join(IMAGES_DIR, 'tutorial_5_spectrogram.png')
if os.path.exists(spectrogram_image):
    st.image(spectrogram_image, caption="Example output: Spectrogram showing frequency content over time", use_container_width=True)

st.markdown("""
**Reading a spectrogram:**
- **Horizontal lines**: Sustained pitches or harmonics
- **Bright spots**: Loud frequency components
- **Vertical stripes**: Transients (attacks, percussive sounds)
- **Harmonic series**: Multiple horizontal lines at regular frequency intervals indicate a pitched instrument
""")

st.header("Step 5: Create a Mel Spectrogram")
st.markdown("""
A **Mel spectrogram** uses a frequency scale that better matches human hearing perception.
We hear the difference between 100 Hz and 200 Hz more easily than between 5000 Hz and 5100 Hz,
even though both are 100 Hz apart. The Mel scale accounts for this.
""")
st.code("""
# Compute the Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert to decibels
S_db = librosa.power_to_db(S, ref=np.max)

# Plot
plt.figure(figsize=(14, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Mel scale)')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
""", language="python")

# Show example output image
mel_spectrogram_image = os.path.join(IMAGES_DIR, 'tutorial_5_mel_spectrogram.png')
if os.path.exists(mel_spectrogram_image):
    st.image(mel_spectrogram_image, caption="Example output: Mel spectrogram with perceptually-scaled frequency axis", use_container_width=True)

st.markdown("""
**Why use Mel spectrograms?**
- Better match human perception of pitch
- Commonly used in machine learning for music classification
- Reduce the amount of data while keeping perceptually relevant information
""")

st.header("Step 6: Load Your Own Audio")
st.markdown("""
You can upload your own audio file to Colab and analyze it. Here's how:
""")
st.code("""
# Upload a file from your computer
from google.colab import files

# This will open a file picker dialog
uploaded = files.upload()

# Get the filename (assuming one file uploaded)
filename = list(uploaded.keys())[0]

# Load the uploaded audio
y_custom, sr_custom = librosa.load(filename)

# Print info and play
print(f"Loaded: {filename}")
print(f"Sample rate: {sr_custom} Hz")
print(f"Duration: {len(y_custom) / sr_custom:.2f} seconds")
Audio(y_custom, rate=sr_custom)
""", language="python")

st.markdown("""
**Tips for uploading audio:**
- Supported formats: WAV, MP3, OGG, FLAC, and more
- Shorter files (under 1 minute) work best for quick experimentation
- librosa will automatically convert stereo to mono unless you specify otherwise
""")

st.header("Complete Example: Side-by-Side Visualizations")
st.markdown("""
Here's a complete script that creates all three visualizations side by side:
""")
st.code("""
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

# Load audio
y, sr = librosa.load(librosa.ex('trumpet'))

# Create a figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.8)
axes[0].set_title('Waveform')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude')

# 2. Spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='magma')
axes[1].set_title('Spectrogram')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

# 3. Mel Spectrogram
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
img2 = librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis')
axes[2].set_title('Mel Spectrogram')
fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

plt.tight_layout()
plt.show()

# Play the audio
Audio(y, rate=sr)
""", language="python")

# Show example output image
combined_image = os.path.join(IMAGES_DIR, 'tutorial_5_combined_visualizations.png')
if os.path.exists(combined_image):
    st.image(combined_image, caption="Example output: Combined visualizations showing waveform, spectrogram, and Mel spectrogram", use_container_width=True)

st.header("Run in Google Colab")
st.markdown("Click below to open a pre-filled Colab notebook with all the code from this tutorial:")
st.link_button("Open in Colab", "https://colab.research.google.com/drive/1XxbQWuk2PnIqRG1lOe5yl2BwlGPHQmlK?usp=sharing")
               

st.header("Tips and Next Steps")
st.markdown("""
- **Experiment**: Try loading different audio files and compare their visualizations
- **Zoom in**: Use matplotlib's interactive zoom to examine details in spectrograms
- **Compare instruments**: Load recordings of different instruments to see how their spectrograms differ
- **Next tutorial**: Learn about beat detection and tempo analysis with librosa!

### Key Concepts Learned:
1. Audio is represented as numerical arrays (samples)
2. Sample rate determines audio resolution and quality
3. Waveforms show amplitude over time
4. Spectrograms reveal frequency content over time
5. Mel spectrograms align with human pitch perception
""")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
