import streamlit as st

st.title("Tutorial: Audio Effects & Exporting with librosa and soundfile")

st.markdown("""
This tutorial teaches you how to apply **audio effects** using librosa and export your processed
audio using **soundfile**. These skills let you manipulate audio programmatically—from changing
tempo to creating special effects—and save your work for use in music production or further analysis.

This tutorial is designed for music students with little to no programming experience.
Run the code in Google Colab for free (no local setup required).
""")

st.header("About Audio Processing and soundfile")
st.markdown("""
### librosa Effects:
librosa provides tools to transform audio in musically meaningful ways:
- **Time stretching**: Change duration without affecting pitch
- **Pitch shifting**: Transpose up or down without affecting tempo
- **Harmonic-Percussive separation**: Isolate melody from rhythm
- **Dynamic range manipulation**: Normalize volume levels

### soundfile Library:
**soundfile** is a Python library for reading and writing audio files. While librosa excels at
analysis, soundfile provides robust file I/O:
- **Format support**: WAV, FLAC, OGG, and more
- **High-quality encoding**: Lossless formats preserve audio fidelity
- **Simple API**: Easy read/write operations
- **Metadata handling**: Preserve or add file metadata

### Applications:
- **Practice tools**: Slow down difficult passages without changing pitch
- **Remixing**: Separate and recombine audio components
- **Sound design**: Create effects and transformations
- **Batch processing**: Process multiple files programmatically
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
Install and import the necessary libraries:
""")
st.code("""
!pip install librosa soundfile

import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import os

print("Libraries loaded successfully!")
print(f"soundfile version: {sf.__version__}")
""", language="python")

st.header("Step 2: Load Audio and Basic Information")
st.markdown("""
Let's start by loading audio and examining its properties:
""")
st.code("""
# Load audio
y, sr = librosa.load(librosa.ex('trumpet'))

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(y) / sr:.2f} seconds")
print(f"Number of samples: {len(y)}")
print(f"Data type: {y.dtype}")

# Listen to the original
print("\\nOriginal audio:")
Audio(y, rate=sr)
""", language="python")

st.header("Step 3: Time Stretching")
st.markdown("""
**Time stretching** changes the duration of audio without altering its pitch.
This is perfect for slowing down fast passages for practice or speeding up
slow music for analysis.

The `rate` parameter controls the stretch:
- `rate > 1.0`: Audio plays faster (shorter duration)
- `rate < 1.0`: Audio plays slower (longer duration)
""")
st.code("""
# Slow down to 75% speed (good for practice!)
y_slow = librosa.effects.time_stretch(y, rate=0.75)

print(f"Original duration: {len(y) / sr:.2f} seconds")
print(f"Slowed duration: {len(y_slow) / sr:.2f} seconds")

# Listen to the slowed version
print("\\nSlowed to 75% speed:")
Audio(y_slow, rate=sr)
""", language="python")

st.code("""
# Speed up to 125% speed
y_fast = librosa.effects.time_stretch(y, rate=1.25)

print(f"Original duration: {len(y) / sr:.2f} seconds")
print(f"Fast duration: {len(y_fast) / sr:.2f} seconds")

# Listen to the faster version
print("\\nSped up to 125% speed:")
Audio(y_fast, rate=sr)
""", language="python")

st.header("Step 4: Pitch Shifting")
st.markdown("""
**Pitch shifting** transposes audio up or down without changing its tempo.
The `n_steps` parameter specifies the number of semitones to shift:
- Positive values: Shift up (higher pitch)
- Negative values: Shift down (lower pitch)
- 12 semitones = 1 octave
""")
st.code("""
# Shift up by 4 semitones (a major third)
y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)

print("Pitch shifted UP by 4 semitones (major third):")
Audio(y_up, rate=sr)
""", language="python")

st.code("""
# Shift down by 5 semitones (a perfect fourth)
y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)

print("Pitch shifted DOWN by 5 semitones (perfect fourth):")
Audio(y_down, rate=sr)
""", language="python")

st.code("""
# Shift down by 12 semitones (one octave)
y_octave_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-12)

print("Pitch shifted DOWN by 12 semitones (one octave):")
Audio(y_octave_down, rate=sr)
""", language="python")

st.header("Step 5: Combining Effects")
st.markdown("""
You can chain multiple effects together. Let's slow down the audio AND
transpose it to help with practice:
""")
st.code("""
# Slow down to 70% AND transpose down 2 semitones
# (useful for practicing a difficult passage in a lower key)
y_practice = librosa.effects.time_stretch(y, rate=0.7)
y_practice = librosa.effects.pitch_shift(y_practice, sr=sr, n_steps=-2)

print(f"Original duration: {len(y) / sr:.2f} seconds")
print(f"Practice version: {len(y_practice) / sr:.2f} seconds")
print("\\nSlowed to 70% and transposed down 2 semitones:")
Audio(y_practice, rate=sr)
""", language="python")

st.header("Step 6: Volume Normalization")
st.markdown("""
**Normalization** adjusts the volume so the loudest peak reaches a target level.
This ensures consistent volume across different audio files.
""")
st.code("""
# Check the current peak amplitude
print(f"Original peak amplitude: {np.max(np.abs(y)):.4f}")

# Normalize to peak at 1.0 (maximum without clipping)
y_normalized = librosa.util.normalize(y)
print(f"Normalized peak amplitude: {np.max(np.abs(y_normalized)):.4f}")

# Normalize to a specific level (e.g., -3 dB below max)
# Convert -3 dB to linear: 10^(-3/20) ≈ 0.708
target_amplitude = 10**(-3/20)
y_normalized_3db = librosa.util.normalize(y, norm=np.inf, threshold=None)
y_normalized_3db = y_normalized_3db * target_amplitude
print(f"Normalized to -3dB: {np.max(np.abs(y_normalized_3db)):.4f}")

Audio(y_normalized, rate=sr)
""", language="python")

st.header("Step 7: Export Audio with soundfile")
st.markdown("""
**soundfile** makes it easy to save your processed audio to various formats.
The most common format is WAV (lossless), but FLAC (lossless, compressed)
and OGG (lossy, small file size) are also popular.
""")
st.code("""
# Export as WAV (uncompressed, lossless)
sf.write('output_processed.wav', y_slow, sr)
print("Saved: output_processed.wav")

# Export as FLAC (compressed, lossless)
sf.write('output_processed.flac', y_slow, sr)
print("Saved: output_processed.flac")

# Check file sizes
wav_size = os.path.getsize('output_processed.wav')
flac_size = os.path.getsize('output_processed.flac')
print(f"\\nWAV file size: {wav_size:,} bytes")
print(f"FLAC file size: {flac_size:,} bytes")
print(f"FLAC is {100 * (1 - flac_size/wav_size):.1f}% smaller")
""", language="python")

st.header("Step 8: Batch Processing Multiple Effects")
st.markdown("""
Here's a practical example: creating multiple practice versions of a piece
at different tempos and transpositions:
""")
st.code("""
# Define practice variations
variations = [
    {'name': 'slow_50', 'rate': 0.5, 'semitones': 0},
    {'name': 'slow_75', 'rate': 0.75, 'semitones': 0},
    {'name': 'normal_down_2', 'rate': 1.0, 'semitones': -2},
    {'name': 'slow_75_down_3', 'rate': 0.75, 'semitones': -3},
]

# Process and save each variation
for var in variations:
    # Apply effects
    y_processed = librosa.effects.time_stretch(y, rate=var['rate'])
    if var['semitones'] != 0:
        y_processed = librosa.effects.pitch_shift(y_processed, sr=sr, n_steps=var['semitones'])
    
    # Normalize
    y_processed = librosa.util.normalize(y_processed)
    
    # Save
    filename = f"practice_{var['name']}.wav"
    sf.write(filename, y_processed, sr)
    print(f"Created: {filename} (rate={var['rate']}, semitones={var['semitones']})")

print("\\nAll practice files created!")
""", language="python")

st.header("Step 9: Read Audio Metadata")
st.markdown("""
soundfile can read metadata from audio files, which is useful for organizing
large audio collections:
""")
st.code("""
# Read file info without loading the audio data
info = sf.info('output_processed.wav')

print("=== Audio File Information ===")
print(f"Duration: {info.duration:.2f} seconds")
print(f"Sample rate: {info.samplerate} Hz")
print(f"Channels: {info.channels}")
print(f"Format: {info.format}")
print(f"Subtype: {info.subtype}")
print(f"Frames (samples): {info.frames}")
""", language="python")

st.header("Step 10: Create a Simple Audio Effect")
st.markdown("""
Let's create some custom effects by manipulating the audio directly:
""")
st.code("""
# Create a fade-in effect (first 1 second)
def apply_fade_in(audio, sr, duration=1.0):
    \"\"\"Apply a fade-in effect to the beginning of audio.\"\"\"
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(audio))  # Don't exceed audio length
    
    # Create fade curve (linear)
    fade_curve = np.linspace(0, 1, fade_samples)
    
    # Apply to audio
    audio_faded = audio.copy()
    audio_faded[:fade_samples] = audio_faded[:fade_samples] * fade_curve
    
    return audio_faded

# Create a fade-out effect (last 1 second)
def apply_fade_out(audio, sr, duration=1.0):
    \"\"\"Apply a fade-out effect to the end of audio.\"\"\"
    fade_samples = int(duration * sr)
    fade_samples = min(fade_samples, len(audio))
    
    fade_curve = np.linspace(1, 0, fade_samples)
    
    audio_faded = audio.copy()
    audio_faded[-fade_samples:] = audio_faded[-fade_samples:] * fade_curve
    
    return audio_faded

# Apply both fades
y_faded = apply_fade_in(y, sr, duration=0.5)
y_faded = apply_fade_out(y_faded, sr, duration=0.5)

# Save the faded version
sf.write('output_with_fades.wav', y_faded, sr)
print("Created audio with fade-in and fade-out")

# Listen
Audio(y_faded, rate=sr)
""", language="python")

st.header("Step 11: Visualize Before and After")
st.markdown("""
Create a comparison visualization of original vs. processed audio:
""")
st.code("""
# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original
librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.7)
axes[0].set_title('Original Audio')

# Time stretched (slow)
librosa.display.waveshow(y_slow, sr=sr, ax=axes[1], alpha=0.7, color='green')
axes[1].set_title('Time Stretched (75% speed)')

# Pitch shifted
librosa.display.waveshow(y_up, sr=sr, ax=axes[2], alpha=0.7, color='purple')
axes[2].set_title('Pitch Shifted (+4 semitones)')

plt.tight_layout()
plt.show()
""", language="python")

st.header("Complete Processing Pipeline")
st.markdown("""
Here's a complete function for processing and exporting audio:
""")
st.code("""
def process_and_export(input_path, output_path, 
                       speed_factor=1.0, 
                       pitch_semitones=0, 
                       normalize=True,
                       fade_in=0, 
                       fade_out=0,
                       output_format='wav'):
    \"\"\"
    Complete audio processing and export function.
    
    Args:
        input_path: Path to input audio file (or librosa example name)
        output_path: Path for output file (without extension)
        speed_factor: Time stretch factor (1.0 = no change)
        pitch_semitones: Pitch shift in semitones (0 = no change)
        normalize: Whether to normalize volume
        fade_in: Fade-in duration in seconds
        fade_out: Fade-out duration in seconds
        output_format: Output format ('wav', 'flac', 'ogg')
    \"\"\"
    # Load audio
    if input_path.startswith('librosa.ex'):
        example_name = input_path.split("'")[1]
        y, sr = librosa.load(librosa.ex(example_name))
    else:
        y, sr = librosa.load(input_path)
    
    # Apply time stretching
    if speed_factor != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_factor)
    
    # Apply pitch shifting
    if pitch_semitones != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)
    
    # Apply fades
    if fade_in > 0:
        y = apply_fade_in(y, sr, fade_in)
    if fade_out > 0:
        y = apply_fade_out(y, sr, fade_out)
    
    # Normalize
    if normalize:
        y = librosa.util.normalize(y)
    
    # Export
    output_file = f"{output_path}.{output_format}"
    sf.write(output_file, y, sr)
    
    return output_file, len(y) / sr

# Example usage
output_file, duration = process_and_export(
    input_path="librosa.ex('trumpet')",
    output_path="my_processed_audio",
    speed_factor=0.8,
    pitch_semitones=-2,
    normalize=True,
    fade_in=0.3,
    fade_out=0.5,
    output_format='wav'
)

print(f"Created: {output_file}")
print(f"Duration: {duration:.2f} seconds")

# Listen to the result
y_result, sr_result = librosa.load(output_file)
Audio(y_result, rate=sr_result)
""", language="python")

st.header("Run in Google Colab")
st.markdown("Click below to open a pre-filled Colab notebook with all the code from this tutorial:")
st.link_button("Open in Colab", "https://colab.research.google.com/")

st.header("Tips and Next Steps")
st.markdown("""
### Best Practices:
- **Preserve originals**: Always work on copies of your audio files
- **Use lossless formats**: WAV or FLAC for intermediate processing
- **Normalize at the end**: Apply normalization as the final step
- **Check for clipping**: Ensure peak amplitude doesn't exceed 1.0

### Practice Exercises:
1. **Create a practice tool**: Make versions at 50%, 75%, and 100% speed
2. **Build a transposition helper**: Generate all 12 transpositions of a melody
3. **Design a crossfade**: Blend two audio files together

### Key Concepts Learned:
1. Time stretching changes duration without affecting pitch
2. Pitch shifting transposes without affecting tempo
3. soundfile provides robust audio file I/O
4. Effects can be chained for complex processing
5. Normalization ensures consistent volume levels
""")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
