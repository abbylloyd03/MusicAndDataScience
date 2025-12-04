import streamlit as st

st.title("Tutorial: Beat Detection & Tempo Analysis with librosa")

st.markdown("""
This tutorial teaches you how to use **librosa** to detect beats and analyze tempo in audio recordings.
Understanding a song's rhythmic structure computationally is essential for music information retrieval,
DJ software, automatic playlist generation, and music production tools.

This tutorial is designed for music students with little to no programming experience.
Run the code in Google Colab for free (no local setup required).
""")

st.header("About Beat Detection")
st.markdown("""
**Beat detection** (also called beat tracking) is the process of identifying the locations of musical
beats in an audio signal. It's one of the most practical applications of music information retrieval.

### Why Beat Detection Matters:
- **DJ Software**: Automatic tempo matching and beat synchronization
- **Music Analysis**: Understanding rhythmic structure and meter
- **Audio Editing**: Quantizing recordings to a tempo grid
- **Machine Learning**: Features for music classification and recommendation

### How It Works:
librosa uses an **onset strength envelope** (which highlights where new sounds begin) combined
with **dynamic programming** to find a tempo that best explains the pattern of onsets. It looks
for regular patterns of emphasis in the audio—the pulse you tap your foot to.
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
First, install and import the necessary libraries:
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

st.header("Step 2: Load Audio and Detect Tempo")
st.markdown("""
librosa provides `librosa.beat.beat_track()` which simultaneously estimates the tempo (BPM)
and identifies the time locations of each beat. Let's try it with a built-in example:
""")
st.code("""
# Load audio (a choice of examples: 'trumpet', 'nutcracker', 'choice', 'brahms')
y, sr = librosa.load(librosa.ex('choice'))

# Detect tempo and beats
# tempo: estimated beats per minute
# beats: frame indices where beats occur
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

print(f"Estimated tempo: {tempo[0]:.1f} BPM")
print(f"Number of beats detected: {len(beats)}")

# Convert beat frames to time (in seconds)
beat_times = librosa.frames_to_time(beats, sr=sr)
print(f"First 5 beat times (seconds): {beat_times[:5]}")

# Listen to the audio
Audio(y, rate=sr)
""", language="python")

st.markdown("""
**Understanding the Output:**
- **tempo**: A single number in beats per minute (BPM)
- **beats**: An array of frame indices where librosa detected beats
- **beat_times**: Converted to seconds for easier interpretation
""")

st.header("Step 3: Visualize Beats on the Waveform")
st.markdown("""
Let's create a visualization that shows exactly where the detected beats fall
on the audio waveform. This helps verify that the beat detection is accurate.
""")
st.code("""
# Create the plot
plt.figure(figsize=(14, 4))

# Plot the waveform
librosa.display.waveshow(y, sr=sr, alpha=0.6)

# Mark each beat with a vertical line
for beat_time in beat_times:
    plt.axvline(x=beat_time, color='red', alpha=0.5, linestyle='--')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title(f'Waveform with Detected Beats (Tempo: {tempo[0]:.1f} BPM)')
plt.tight_layout()
plt.show()
""", language="python")

st.header("Step 4: Create a Click Track")
st.markdown("""
One of the most useful applications of beat detection is generating a **click track**—
audio clicks at each detected beat. This lets you hear whether the beat detection
matches your perception of the pulse. We'll use `librosa.clicks()` to generate click sounds.
""")
st.code("""
# Generate click sounds at each beat time
clicks = librosa.clicks(times=beat_times, sr=sr, length=len(y))

# Mix the clicks with the original audio
# Reduce click volume to 0.3 so it doesn't overpower the music
y_with_clicks = y + 0.3 * clicks

# Listen to the result
print("Listen to hear clicks on each detected beat:")
Audio(y_with_clicks, rate=sr)
""", language="python")

st.markdown("""
**Evaluating Beat Detection:**
Listen carefully! Do the clicks align with where you perceive the beat?
If not, the tempo might be doubled or halved, or the audio might have
an unusual rhythm that challenges the algorithm.
""")

st.header("Step 5: Analyze the Onset Strength Envelope")
st.markdown("""
The **onset strength envelope** shows how much musical activity (energy changes)
occurs at each moment. It's the foundation for beat detection—peaks often
correspond to beats. Visualizing it helps understand why certain beats were detected.
""")
st.code("""
# Compute the onset strength envelope
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# Create time axis for plotting
times = librosa.times_like(onset_env, sr=sr)

# Plot the onset strength envelope with beats
plt.figure(figsize=(14, 4))
plt.plot(times, onset_env, label='Onset Strength')

# Mark the beat locations
for beat_time in beat_times:
    plt.axvline(x=beat_time, color='red', alpha=0.5, linestyle='--')

plt.xlabel('Time (seconds)')
plt.ylabel('Onset Strength')
plt.title('Onset Strength Envelope with Detected Beats')
plt.legend()
plt.tight_layout()
plt.show()
""", language="python")

st.markdown("""
**Reading the Onset Envelope:**
- **Peaks**: Strong onsets (attacks, loud notes)
- **Valleys**: Sustained notes or silence
- **Regular spacing**: Indicates steady tempo
- **Beat lines**: Should align with major peaks for accurate detection
""")

st.header("Step 6: Create a Tempogram")
st.markdown("""
A **tempogram** shows how the tempo varies over time. It's a 2D visualization
where the x-axis is time, the y-axis is tempo (BPM), and brightness indicates
how strongly that tempo is present at each moment. This is useful for music
with tempo changes or rubato (expressive timing).
""")
st.code("""
# Compute the tempogram
# hop_length controls time resolution
tempogram = librosa.feature.tempogram(y=y, sr=sr)

# Plot the tempogram
plt.figure(figsize=(14, 6))
librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo', cmap='magma')
plt.colorbar(label='Tempo Strength')
plt.axhline(y=tempo[0], color='white', linestyle='--', alpha=0.7, label=f'Estimated tempo: {tempo[0]:.1f} BPM')
plt.title('Tempogram: Tempo Estimation Over Time')
plt.legend()
plt.tight_layout()
plt.show()
""", language="python")

st.markdown("""
**Reading a Tempogram:**
- **Bright horizontal lines**: Consistent tempo throughout
- **Multiple lines**: Could indicate tempo harmonics (double/half time)
- **Wandering brightness**: Tempo changes or rubato
- **Vertical patterns**: Rhythmic events affecting multiple tempo bands
""")

st.header("Step 7: Compare Original and Detected Tempo")
st.markdown("""
Here's a complete analysis that combines all visualizations and
exports the beat times for use in other applications:
""")
st.code("""
import pandas as pd

# Comprehensive beat analysis
def analyze_beats(audio_path_or_array, sr=None):
    # Load audio if path provided
    if isinstance(audio_path_or_array, str):
        y, sr = librosa.load(audio_path_or_array)
    else:
        y = audio_path_or_array
    
    # Detect tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Calculate beat intervals (time between beats)
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        avg_interval = np.mean(beat_intervals)
        interval_std = np.std(beat_intervals)
    else:
        beat_intervals = []
        avg_interval = 0
        interval_std = 0
    
    # Create summary
    summary = {
        'Estimated Tempo (BPM)': tempo[0],
        'Total Beats': len(beats),
        'Duration (seconds)': len(y) / sr,
        'Average Beat Interval (s)': avg_interval,
        'Interval Std Dev (s)': interval_std,
        'Tempo Consistency (%)': 100 * (1 - interval_std / avg_interval) if avg_interval > 0 else 0
    }
    
    # Create DataFrame of beat times
    beats_df = pd.DataFrame({
        'Beat Number': range(1, len(beat_times) + 1),
        'Time (seconds)': beat_times
    })
    
    return summary, beats_df, y, sr, beat_times

# Run the analysis
summary, beats_df, y, sr, beat_times = analyze_beats(y, sr)

# Print summary
print("=== Beat Analysis Summary ===")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Show first few beats
print("\\n=== First 10 Beats ===")
print(beats_df.head(10))

# Export to CSV
beats_df.to_csv('detected_beats.csv', index=False)
print("\\nBeat times saved to 'detected_beats.csv'")
""", language="python")

st.header("Step 8: Adjust Beat Detection Parameters")
st.markdown("""
librosa allows fine-tuning beat detection. Key parameters include:
- **start_bpm**: Initial tempo estimate (default: 120)
- **tightness**: How tightly beats should follow tempo (higher = stricter)
- **trim**: Whether to ignore silent sections at start/end
""")
st.code("""
# Example: Adjust beat detection for faster music
tempo_fast, beats_fast = librosa.beat.beat_track(
    y=y, 
    sr=sr,
    start_bpm=140,    # Start with higher tempo estimate
    tightness=200     # Allow more tempo flexibility (default is 100)
)

print(f"With start_bpm=140: {tempo_fast[0]:.1f} BPM, {len(beats_fast)} beats")

# Compare with default
tempo_default, beats_default = librosa.beat.beat_track(y=y, sr=sr)
print(f"Default settings:   {tempo_default[0]:.1f} BPM, {len(beats_default)} beats")
""", language="python")

st.header("Run in Google Colab")
st.markdown("Click below to open a pre-filled Colab notebook with all the code from this tutorial:")
st.link_button("Open in Colab", "https://colab.research.google.com/")

st.header("Tips and Next Steps")
st.markdown("""
### Common Issues and Solutions:
- **Tempo doubled/halved**: The algorithm sometimes finds double or half the actual tempo.
  Use `start_bpm` to guide it toward the expected range.
- **Missed beats**: Increase `tightness` for steadier tempo tracking.
- **Complex rhythms**: Some music (rubato, complex meters) challenges all beat trackers.

### Practice Exercises:
1. **Compare genres**: Try music from different genres and compare beat detection accuracy
2. **Create a metronome**: Generate a pure click track at the detected tempo
3. **Analyze tempo changes**: Use the tempogram to find songs with tempo variations

### Key Concepts Learned:
1. Beat tracking uses onset strength to find rhythmic pulse
2. Tempo is measured in beats per minute (BPM)
3. Click tracks help verify beat detection accuracy
4. Onset envelopes show musical energy over time
5. Tempograms reveal tempo consistency and changes
""")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
