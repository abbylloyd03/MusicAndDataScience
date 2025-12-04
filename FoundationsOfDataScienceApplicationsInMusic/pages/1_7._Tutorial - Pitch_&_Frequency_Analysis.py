import streamlit as st

st.title("Tutorial: Pitch & Frequency Analysis with librosa")

st.markdown("""
This tutorial explores how to analyze **pitch** and **frequency content** in audio using librosa.
Understanding the harmonic content of audio is fundamental to computational musicology,
automatic transcription, and instrument recognition.

This tutorial is designed for music students with little to no programming experience.
Run the code in Google Colab for free (no local setup required).
""")

st.header("About Pitch and Frequency Analysis")
st.markdown("""
When we hear a musical pitch, we're actually perceiving the **fundamental frequency** of a sound
along with its **harmonic overtones**. A violin playing A4 and a piano playing A4 both produce
a fundamental frequency of 440 Hz, but they sound different because of their unique overtone patterns.

### Key Concepts:
- **Pitch**: The perceptual quality of how "high" or "low" a note sounds
- **Fundamental Frequency (F0)**: The lowest frequency component that defines the perceived pitch
- **Harmonics**: Integer multiples of the fundamental that give instruments their timbre
- **Chromagram**: A representation showing the intensity of each pitch class (C, C#, D, etc.) over time

### Applications:
- **Automatic Music Transcription**: Converting audio to musical notation
- **Key Detection**: Identifying the musical key of a recording
- **Melody Extraction**: Isolating the main melodic line
- **Instrument Identification**: Recognizing instruments by their harmonic content
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
Install and import the necessary libraries:
""")
st.code("""
!pip install librosa soundfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
print("Libraries loaded successfully!")
""", language="python")

st.header("Step 2: Load Audio and Visualize the Spectrum")
st.markdown("""
The **frequency spectrum** shows the intensity of each frequency at a given moment.
Unlike a spectrogram (which shows frequencies over time), a single spectrum captures
one instant—like a snapshot of the audio's harmonic content.
""")
st.code("""
# Load audio
y, sr = librosa.load(librosa.ex('trumpet'))

# Listen first
print("Listen to the audio:")
Audio(y, rate=sr)
""", language="python")

st.code("""
# Compute the Short-Time Fourier Transform
D = librosa.stft(y)

# Take the magnitude of one frame (middle of the audio)
frame_idx = len(D[0]) // 2
spectrum = np.abs(D[:, frame_idx])

# Create frequency axis
frequencies = librosa.fft_frequencies(sr=sr)

# Plot the spectrum
plt.figure(figsize=(14, 4))
plt.plot(frequencies, spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum (Single Frame)')
plt.xlim(0, 5000)  # Focus on 0-5000 Hz
plt.tight_layout()
plt.show()
""", language="python")

st.markdown("""
**Reading a spectrum:**
- The **first peak** (leftmost) is usually the fundamental frequency
- **Higher peaks** are harmonics (integer multiples of the fundamental)
- The pattern of peak heights determines the instrument's timbre
""")

st.header("Step 3: Create a Chromagram")
st.markdown("""
A **chromagram** (also called a pitch class profile) shows the energy in each of the 12 pitch classes
(C, C#, D, D#, E, F, F#, G, G#, A, A#, B) over time. It collapses all octaves together—so all C notes
(C2, C3, C4, etc.) contribute to the same "C" row.

This is incredibly useful for:
- Chord detection and harmonic analysis
- Key detection
- Cover song identification
""")
st.code("""
# Compute the chromagram
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Plot
plt.figure(figsize=(14, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='coolwarm')
plt.colorbar(label='Intensity')
plt.title('Chromagram - Pitch Class Energy Over Time')
plt.tight_layout()
plt.show()
""", language="python")

st.markdown("""
**Reading a chromagram:**
- **Bright horizontal bands**: Strong presence of that pitch class
- **Vertical patterns**: Chord changes (multiple pitch classes active simultaneously)
- **Constant brightness in one row**: A sustained note or pedal tone
""")

st.header("Step 4: Pitch Detection with PYIN")
st.markdown("""
**PYIN** (Probabilistic YIN) is a robust algorithm for detecting the fundamental frequency
(pitch) of a monophonic audio signal (single melody line). It returns the estimated pitch
in Hertz for each time frame.
""")
st.code("""
# Detect pitch using PYIN
# f0: fundamental frequency in Hz (NaN where no pitch detected)
# voiced_flag: True where voice/pitch is detected
# voiced_probs: probability of voicing at each frame
f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                               fmin=librosa.note_to_hz('C2'),
                                               fmax=librosa.note_to_hz('C7'))

# Create time axis
times = librosa.times_like(f0)

# Plot the detected pitch
plt.figure(figsize=(14, 4))
plt.plot(times, f0, label='Detected F0', color='blue')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch Detection using PYIN')
plt.legend()
plt.tight_layout()
plt.show()

# Print some statistics
valid_f0 = f0[~np.isnan(f0)]
if len(valid_f0) > 0:
    print(f"Average detected pitch: {np.mean(valid_f0):.1f} Hz")
    print(f"Pitch range: {np.min(valid_f0):.1f} - {np.max(valid_f0):.1f} Hz")
""", language="python")

st.header("Step 5: Convert Frequencies to Musical Notes")
st.markdown("""
librosa can convert between frequencies (Hz) and musical note names.
This makes pitch analysis results more musically meaningful.
""")
st.code('''
# Convert detected pitches to note names
def hz_to_note_name(freq):
    """Convert frequency to note name, handling NaN values."""
    if np.isnan(freq):
        return None
    return librosa.hz_to_note(freq)

# Get unique notes detected
valid_pitches = f0[~np.isnan(f0)]
note_names = [hz_to_note_name(freq) for freq in valid_pitches]

# Count occurrences of each note
from collections import Counter
note_counts = Counter(note_names)

print("Most common notes detected:")
for note, count in note_counts.most_common(10):
    print(f"  {note}: {count} frames")
''', language="python")

st.header("Step 6: Overlay Pitch on Spectrogram")
st.markdown("""
Visualizing the detected pitch on top of a spectrogram helps verify the accuracy
of pitch detection. The pitch line should follow the brightest harmonic contours.
""")
st.code("""
# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

# Plot spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')

# Overlay detected pitch
times = librosa.times_like(f0)
ax.plot(times, f0, color='cyan', linewidth=2, label='Detected Pitch')

ax.set_ylim(0, 2000)  # Focus on lower frequencies
ax.legend(loc='upper right')
ax.set_title('Spectrogram with Detected Pitch Overlay')
plt.colorbar(ax.images[0], ax=ax, format='%+2.0f dB')
plt.tight_layout()
plt.show()
""", language="python")

st.header("Step 7: Harmonic-Percussive Source Separation")
st.markdown("""
Music contains both **harmonic** (pitched) and **percussive** (rhythmic) components.
librosa can separate these, which is useful for:
- Isolating melody for pitch analysis
- Extracting drums for beat detection
- Remixing and audio effects
""")
st.code("""
# Separate harmonic and percussive components
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Create visualizations
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original
librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.7)
axes[0].set_title('Original Audio')

# Harmonic
librosa.display.waveshow(y_harmonic, sr=sr, ax=axes[1], alpha=0.7, color='blue')
axes[1].set_title('Harmonic Component (Pitched sounds)')

# Percussive
librosa.display.waveshow(y_percussive, sr=sr, ax=axes[2], alpha=0.7, color='red')
axes[2].set_title('Percussive Component (Transients, drums)')

plt.tight_layout()
plt.show()

# Listen to the separated components
print("Original:")
Audio(y, rate=sr)
""", language="python")

st.code("""
print("Harmonic component only:")
Audio(y_harmonic, rate=sr)
""", language="python")

st.code("""
print("Percussive component only:")
Audio(y_percussive, rate=sr)
""", language="python")

st.header("Step 8: Key Detection")
st.markdown("""
By analyzing the chromagram, we can estimate the musical **key** of a piece.
This involves comparing the chroma distribution to templates for major and minor keys.
""")
st.code("""
# Compute chromagram
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Sum across time to get overall pitch class distribution
chroma_sum = np.sum(chroma, axis=1)

# Normalize
chroma_sum = chroma_sum / np.max(chroma_sum)

# Pitch class names
pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Plot pitch class distribution
plt.figure(figsize=(10, 4))
plt.bar(pitch_classes, chroma_sum, color='steelblue')
plt.xlabel('Pitch Class')
plt.ylabel('Normalized Energy')
plt.title('Pitch Class Distribution')
plt.tight_layout()
plt.show()

# Find the most prominent pitch class
dominant_pc = pitch_classes[np.argmax(chroma_sum)]
print(f"Most prominent pitch class: {dominant_pc}")
print("(This suggests the piece may be in or related to this key)")
""", language="python")

st.header("Complete Analysis Example")
st.markdown("""
Here's a complete script that performs comprehensive pitch analysis:
""")
st.code('''
import pandas as pd

def analyze_pitch(y, sr):
    """Comprehensive pitch analysis function."""
    
    results = {}
    
    # 1. Detect pitch with PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    valid_f0 = f0[~np.isnan(f0)]
    
    if len(valid_f0) > 0:
        results['avg_pitch_hz'] = np.mean(valid_f0)
        results['pitch_range_hz'] = (np.min(valid_f0), np.max(valid_f0))
        results['avg_note'] = librosa.hz_to_note(results['avg_pitch_hz'])
    
    # 2. Compute chromagram and find dominant pitch class
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    results['dominant_pitch_class'] = pitch_classes[np.argmax(chroma_sum)]
    
    # 3. Harmonic-Percussive ratio
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm**2)
    perc_energy = np.sum(y_perc**2)
    results['harmonic_ratio'] = harm_energy / (harm_energy + perc_energy + 1e-10)
    
    return results

# Run analysis
results = analyze_pitch(y, sr)

print("=== Pitch Analysis Results ===")
for key, value in results.items():
    print(f"{key}: {value}")
''', language="python")

st.header("Run in Google Colab")
st.markdown("Click below to open a pre-filled Colab notebook with all the code from this tutorial:")
st.link_button("Open in Colab", "https://colab.research.google.com/")

st.header("Tips and Next Steps")
st.markdown("""
### Understanding the Results:
- **PYIN works best on monophonic audio** (single melody). For polyphonic music (chords), use chromagrams.
- **Harmonic/Percussive separation** is useful preprocessing for many analysis tasks.
- **Key detection** is a complex problem—the simple approach here gives hints but isn't foolproof.

### Practice Exercises:
1. **Compare instruments**: Analyze the same melody played on different instruments
2. **Transpose detection**: Load two versions of a song and compare their chromagrams
3. **Melody extraction**: Use HPSS to isolate the harmonic content, then run pitch detection

### Key Concepts Learned:
1. Frequency spectrum shows harmonic content at one moment
2. Chromagrams show pitch class energy over time (octave-independent)
3. PYIN detects fundamental frequency in monophonic audio
4. Harmonic-Percussive separation isolates pitched and rhythmic elements
5. Pitch class distribution can hint at musical key
""")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
