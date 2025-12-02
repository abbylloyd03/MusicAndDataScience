import streamlit as st

st.title("1. Tutorial: Hello Melody with music21")

st.markdown("""
This is your "Hello World" for music programming—a simple lab where you'll generate a basic melody using music21, an open-source Python library. This tutorial is designed for music students experimenting with code for the first time. No programming knowledge is needed.

Run this in Google Colab for free (no local setup). Copy the code below into a new Colab notebook, or use the button for a pre-filled one.
""")

st.header("About music21")
st.markdown("""
music21 is an open-source toolkit developed at MIT for computational musicology and symbolic music data. It provides a powerful set of tools for working with music in Python, making it ideal for both beginners and advanced users in music programming.

### Key Capabilities:
- **Creation and Manipulation**: Build musical scores by creating notes, chords, streams, and parts. Easily define pitches, durations, keys, tempos, and time signatures.
- **Analysis**: Perform music theory tasks like identifying keys, intervals, chords, and meters. It supports complex analyses such as metrical hierarchies, beaming, and feature extraction for style or composer identification.
- **Format Support**: Import and export in various formats including MIDI, MusicXML, LilyPond, and Humdrum. Integrate with external renderers like MuseScore for sheet music visualization.
- **Visualization and Output**: Generate graphs (with Matplotlib), render sheet music, and play back audio via MIDI or synthesized WAV files.
- **Integration**: Works seamlessly with Python ecosystems, including libraries like NumPy for data processing or PyFluidSynth for better audio rendering.

For beginners, start with simple tasks like composing melodies (as in this tutorial) or analyzing short pieces. music21 encourages exploration of music through code, from generating basic tunes to advanced computational studies. Check the [official documentation](https://www.music21.org/music21docs/) for more.
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
In Colab, install music21 and additional tools for better audio playback (FluidSynth) in code cells:
""")
st.code("""
!pip install music21
!apt install fluidsynth -y
!pip install pyfluidsynth
""", language="bash")

st.header("Step 2: Create and Play Your 'Hello Melody'")
st.markdown("""
This code sets a key (C major), tempo (90 BPM), and time signature (4/4). It then builds a simple melody. Edit the `melody_notes` list to change pitches, durations, or add notes!

Note: For accurate tempo playback, we export to MIDI and render to WAV using FluidSynth, as the basic MIDI player may ignore tempo changes.
""")
st.code("""
from music21 import *
from IPython.display import Audio

# Define the key, tempo, and time signature
key_sig = key.Key('C')
tempo_mark = tempo.MetronomeMark(number=90)
time_sig = meter.TimeSignature('4/4')

# Create the main score stream
main_stream = stream.Score()

# Create a part for the melody (e.g., piano)
melody_part = stream.Part()
melody_part.insert(0, instrument.Piano())

# Define notes for your "Hello Melody" (edit here!)
melody_notes = [
    note.Note("C4", quarterLength=0.5),  # Hello...
    note.Note("C4", quarterLength=0.5),
    note.Note("G4", quarterLength=0.5),
    note.Note("G4", quarterLength=0.5),
    note.Note("A4", quarterLength=0.5),
    note.Note("A4", quarterLength=0.5),
    note.Note("G4", quarterLength=1.0)   # ...Melody!
]

# Add notes to the melody part
for n in melody_notes:
    melody_part.append(n)

# Add everything to the main stream
main_stream.insert(0, key_sig)
main_stream.insert(0, tempo_mark)
main_stream.insert(0, time_sig)
main_stream.insert(0, melody_part)

# Export the stream to a temporary MIDI file
main_stream.write('midi', fp='temp.mid')

# Render the MIDI to WAV using FluidSynth (respects tempo)
!fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 temp.mid -F temp.wav -r 44100

# Play the audio in Colab
Audio('temp.wav')
""", language="python")

st.markdown("""
Editing Tips: Swap note names (e.g., 'C4' to 'G4' for a different pitch). Change `quarterLength` for rhythm (1.0 = quarter note, 0.5 = faster). Add rests with `note.Rest(quarterLength=1.0)`. Experiment to say "hello" in your musical style!
""")

st.header("Run in Google Colab")
st.markdown("Click to open the updated pre-filled Colab notebook (edit and run there).")
st.link_button("Open in Colab", "https://colab.research.google.com/drive/1o_aMtfiWU1-elLQH4zQas4_z74WGU4kH?usp=sharing")

st.header("Tips and Next Steps")
st.markdown("""
- Why "Hello Melody"? It's like printing "Hello World" in code, but you hear it! Builds skills in music + programming.
- Fun Challenge: Extend the melody to the full "Twinkle Twinkle Little Star" or code your own tune!
- Next Steps:

""")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")