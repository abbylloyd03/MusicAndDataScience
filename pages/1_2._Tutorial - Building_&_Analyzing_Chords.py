import streamlit as st

st.title("Tutorial - Building & Analyzing Chords with music21 and pandas")

st.markdown("""
This lab introduces chords in music21. You'll learn to create, manipulate, and analyze basic chords—groups of multiple pitches played together. Edit the chord pitches to experiment with different harmonies!

Run this in Google Colab for free (no local setup). Copy the code below into a new Colab notebook, or use the button for a pre-filled one.
""")

st.header("About music21 and Chords")
st.markdown("""
music21 is an open-source toolkit developed at MIT for computational musicology and symbolic music data. It provides powerful tools for working with music in Python, including chords, which are objects combining multiple pitches.

### Key Concepts:
- **Creation**: Build chords from pitch names, notes, or MIDI numbers.
- **Manipulation**: Add/remove pitches, set durations, reposition (e.g., closedPosition).
- **Analysis**: Check triad types (major/minor), get common names, inversions, interval vectors.
- **Post-Tonal Features**: Interval vectors, prime forms, Forte classes for advanced analysis.
- **Display/Export**: Show notation, add to streams, export to MIDI.

For beginners, focus on creating simple triads, checking their types, and playing them back. 
""")

st.header("About music21 Chords and pandas Integration")
st.markdown("""
music21 handles chords as objects with pitches, durations, and analytical properties (e.g., commonName for triad type). A Stream is a container for musical elements like chords in sequence.

By extracting data from a Stream into a pandas DataFrame:
- **Storage**: Organize chords in rows with columns for offset, duration, pitches, type, etc.
- **Analysis**: Compute stats (e.g., average duration), filter by type (e.g., major triads), or visualize distributions.
- **Capabilities**: Combine with Pandas features like grouping, sorting, or exporting to CSV for further use in tools like Excel.

This bridges symbolic music representation with data science. For more on Music21 chords, see [Chapter 7](https://www.music21.org/music21docs/usersGuide/usersGuide_07_chords.html).
""")

st.header("Step 1: Setup in Colab")
st.markdown("""
In Colab, install music21, pandas (for DataFrame analysis), and tools for audio playback (FluidSynth) and notation (MuseScore, optional):
""")
st.code("""
!pip install music21 pandas
!apt install fluidsynth -y
!pip install pyfluidsynth
# Optional for notation display:
!apt-get update -qq -y
!apt-get install musescore3 xvfb -y
!Xvfb :99 -screen 0 1024x768x24 &
import os
os.environ['DISPLAY'] = ':99'
from music21 import environment
e = environment.Environment()
e['musescoreDirectPNGPath'] = '/usr/bin/musescore3'
e['pdfPath'] = '/usr/bin/musescore3'
e['graphicsPath'] = '/usr/bin/musescore3'
e['musicxmlPath'] = '/usr/bin/musescore3'
""", language="bash")

st.header("Step 2: Create, Manipulate, and Analyze a Chord")
st.markdown("""
This code creates a C minor triad, sets its duration, checks if it's a minor triad, gets its common name, and plays it back. Edit the `pitches` list to try different chords (e.g., ["C4", "E4", "G4"] for C major)!

Note: Use FluidSynth for accurate playback. Uncomment `cMinor.show()` for notation if MuseScore is set up.
""")
st.code("""
from music21 import *
from IPython.display import Audio

# Create a chord from pitch names (edit here!)
cMinor = chord.Chord(["C4", "E-4", "G4"])

# Set duration
cMinor.duration.type = 'half'  # 2 quarter lengths

# Manipulate: Add a pitch and reposition for display
cMinor.add("B-5")  # Make it a minor seventh chord
cMinor.closedPosition(inPlace=True)  # Stack pitches closely

# Analyze
is_minor_triad = cMinor.isMinorTriad()  # True/False
common_name = cMinor.commonName  # e.g., 'minor triad'
inversion = cMinor.inversion()  # 0 for root position
interval_vector = cMinor.intervalVector  # Post-tonal analysis

# Print analysis (run to see results)
print(f"Is minor triad: {is_minor_triad}")
print(f"Common name: {common_name}")
print(f"Inversion: {inversion}")
print(f"Interval vector: {interval_vector}")

# Optional: Display notation (requires MuseScore setup)
# cMinor.show()

# Export to MIDI and render to WAV for playback
cMinor.write('midi', fp='temp.mid')
!fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 temp.mid -F temp.wav -r 44100

# Play the audio
Audio('temp.wav')
""", language="python")

st.markdown("""
Editing Tips: Change pitch names (e.g., add sharps/flats like 'F#4' or 'B-4'). Try creating a major triad or seventh chord. Use `cMinor.remove("E-4")` to remove pitches. Experiment with inversions by reordering pitches (e.g., ["E-4", "G4", "C5"] for first inversion).
""")

st.header("Step 3: Create a Chord Progression in a Stream")
st.markdown("""
Now, let's build a simple chord progression (e.g., I-IV-V-I in C major) by creating multiple chords and adding them to a Music21 Stream. A Stream is a container for musical elements in sequence. Edit the chords or durations to customize the progression!

This code also inserts a key signature for contextual analysis in the next step.
""")
st.code("""
from music21 import *

# Create a Stream
progression_stream = stream.Stream()

# Define a key (for roman numeral analysis later)
key_sig = key.Key('C')  # C major
progression_stream.insert(0, key_sig)

# Create chords for the progression (edit here!)
chords = [
    chord.Chord(["C4", "E4", "G4"]),  # I (C major)
    chord.Chord(["F4", "A4", "C5"]),  # IV (F major)
    chord.Chord(["G4", "B4", "D5"]),  # V (G major)
    chord.Chord(["C4", "E4", "G4"])   # I (C major)
]

# Add chords to the Stream with durations
for ch in chords:
    ch.duration.quarterLength = 2.0  # Half note each; edit individually if needed
    progression_stream.append(ch)

# Optional: Display the full progression notation (requires MuseScore setup)
# progression_stream.show()

# Playback the progression
progression_stream.write('midi', fp='progression.mid')
!fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 progression.mid -F progression.wav -r 44100
Audio('progression.wav')
""", language="python")

st.markdown("""
Editing Tips: Add more chords (e.g., for ii-V-I jazz progression). Vary durations (e.g., `ch.duration.quarterLength = 1.0` for quarter notes). Insert tempo with `progression_stream.insert(0, tempo.MetronomeMark(number=120))`.
""")

st.header("Step 4: Analyze the Chord Progression Stream")
st.markdown("""
With the progression in a Stream, we can analyze each chord in context (e.g., roman numerals relative to the key). This code iterates through the chords, analyzes them, and prints results like roman figures and inversions.

Edit to add more analysis (e.g., interval vectors for each chord).
""")
st.code("""
from music21 import *

# Assuming 'progression_stream' from Step 3

# Analyze each chord in the stream
for element in progression_stream.getElementsByClass('Chord'):
    # Get roman numeral (requires key context)
    rn = roman.romanNumeralFromChord(element, progression_stream.keySignature)
    
    # Other analyses
    common_name = element.commonName
    inversion = element.inversion()
    
    # Print results
    print(f"Chord at offset {element.offset}: Pitches = {element.pitches}")
    print(f"Roman Numeral: {rn.figure}")
    print(f"Common Name: {common_name}")
    print(f"Inversion: {inversion}")
    print("---")
""", language="python")

st.markdown("""
Analysis Tips: Roman numerals show function (e.g., 'I' for tonic). For minor keys, use `key.Key('c')` (lowercase for minor). Add post-tonal analysis with `element.intervalVector`.
""")

st.header("Step 5: Store the Analysis as a DataFrame")
st.markdown("""
To organize and further analyze the results, store the chord data in a Pandas DataFrame. This code collects analysis info (e.g., offset, pitches, roman numeral, common name) into a DataFrame and displays it. You can then perform operations like filtering or exporting to CSV.

This builds on the analysis from Step 4.
""")
st.code("""
import pandas as pd
from music21 import *

# Assuming 'progression_stream' from Step 3

# Collect analysis data
data = []
for element in progression_stream.getElementsByClass('Chord'):
    rn = roman.romanNumeralFromChord(element, progression_stream.keySignature)
    pitches = [p.nameWithOctave for p in element.pitches]
    data.append({
        'Offset': element.offset,
        'Duration': element.duration.quarterLength,
        'Pitches': ', '.join(pitches),
        'Roman Numeral': rn.figure,
        'Common Name': element.commonName,
        'Inversion': element.inversion(),
        'Is Major Triad': element.isMajorTriad()
    })

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print("Chord Analysis DataFrame:")
print(df)

# Optional: Basic pandas operations (e.g., filter major triads)
major_triads = df[df['Is Major Triad'] == True]
print("\\nMajor Triads:")
print(major_triads)

# Export to CSV
df.to_csv('chord_analysis.csv', index=False)
""", language="python")

st.markdown("""
DataFrame Tips: Add more columns (e.g., 'Interval Vector': element.intervalVector). Use Pandas for stats like `df['Duration'].mean()` or plotting (requires Matplotlib: `!pip install matplotlib` and `df.plot()`).
""")

st.header("Run in Google Colab")
st.markdown("Click to open a pre-filled Colab notebook (edit and run there).")
st.link_button("Open in Colab", "https://colab.research.google.com/drive/1e3N2vqn0lwgi7B3PnjCbFemsqUxFh8Yg?usp=sharing")

st.header("Tips and Next Steps")
st.markdown("""
- **Fun Challenge**: Extend the progression (e.g., add vi-ii-V-I), analyze transitions, or export to MusicXML for import into notation software.
- **Resources**: [music21 Chords Chapter](https://www.music21.org/music21docs/usersGuide/usersGuide_07_chords.html) for advanced topics like Z-relations or Forte classes.

Build on this to explore more music21 features—next, try chordify in Chapter 9!
""")