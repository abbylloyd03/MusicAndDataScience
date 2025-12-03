import streamlit as st

st.title("Tutorial: Chordify & Key Inference with music21")

st.markdown("""
This tutorial explains two common music21 workflows that are useful for harmonic
feature extraction, symbolic analysis, and dataset creation:

- **Chordify** — collapse multi-part scores into a sequence of chord events that
  represent vertical sonorities at each onset.
- **Key inference** — infer a global key for an entire piece and estimate
  local keys (for measures or windows) to support context-aware Roman numeral
  analysis.

This page complements the notebook "4. Chordify and Key Inference" included
in the repository (see the notebooks folder). The notebook contains runnable
examples you can copy into Google Colab.
""")

st.header("What chordify does (summary)")
st.markdown("""
- chordify() is a stream method that collapses simultaneous sounding pitch events
  from all parts into Chord objects placed in a single stream. Example:

  - If two parts contain C4 (melody) and E3 (bass) at the same offset, chordify
    produces a Chord containing both pitches at that offset.

- Typical uses:
  - Produce a harmonic skeleton for Roman numeral labeling.
  - Extract chord-level features (root, quality, inversion, pitch set) for
    machine learning.
  - Quickly audition the vertical harmony by writing the chordified stream to MIDI.

- Caveats:
  - chordify is a heuristic. It simply groups notes that share an offset; it
    does not attempt to model perceived harmony for arpeggios, independent
    inner voices, or staggered entrances. Use careful voice-leading inspection
    if you need a more perceptually faithful reduction.
  - The voicing/order of pitch-class members is determined by the notes in the
    incoming parts and can affect chord.root() and commonName() results.
""")

st.header("Key inference in music21 (summary)")
st.markdown("""
- The convenience method `stream.analyze('key')` (or `score.analyze('key')`) runs
  music21's key-finding procedure and returns a Key object (e.g., `C major` or
  `A minor`). The default algorithm used by music21 is the Krumhansl–Schmuckler
  type profile match (often referenced in music21 as a Krumhansl-based analyzer).
  This compares pitch/PC distributions to idealized profiles to select the best
  tonic and mode.

- Use cases:
  - Get a single global key for the whole piece for a coarse Roman-numeral context.
  - Run `analyze('key')` on substreams (measures, sliding windows) to estimate
    local keys and harmonic movement.

- Caveats:
  - Short windows produce unstable results (too little pitch material). For local
    key inference prefer windows that include several beats/measures (e.g.,
    1–4 measures, depending on style).
  - The algorithm is distributional: heavy use of non-harmonic tones, modal
    borrowings, or tonal ambiguity can lead to surprising outcomes. Validate
    with ear or alternate analyses when in doubt.
""")

st.header("Quick code examples (copy to notebook / Colab)")
st.markdown("Below are short snippets illustrating typical workflows. Copy the cells into a Colab notebook to run them.")

st.header("Install / setup (optional)")
st.code("""
# Optional: install dependencies when running in a fresh environment (uncomment in Colab)
!pip install --quiet music21 pandas pyfluidsynth
!apt-get update -qq && apt-get install -y -qq fluidsynth

print('Skip installs in this environment; ensure music21 and pandas are available.')
""", language="bash")

st.header("Imports")
st.code("""
from music21 import stream, note, chord, roman, key, instrument, meter, tempo, pitch
import pandas as pd
import os
from IPython.display import Audio, display
""", language="python")

st.header("Helper: ensure measures exist")
st.markdown("We build the demo score with explicit measures, but this helper can be used if additional measure creation is required.")
st.code("""
def ensure_score_has_measures(s):
    \"\"\"Ensure that flattened note/chord elements have a measureNumber attribute.
    Returns True if measureNumber attributes are present after attempts; False otherwise.
    \"\"\"
    from music21 import note, chord
    def has_measure_numbers(s2):
        for el in s2.flatten().getElementsByClass([note.Note, chord.Chord]):
            if getattr(el, 'measureNumber', None) is not None:
                return True
        return False

    if has_measure_numbers(s):
        return True

    try:
        if hasattr(s, 'parts') and len(s.parts) > 0:
            for p in s.parts:
                try:
                    p.makeMeasures(inPlace=True)
                except TypeError:
                    ms = p.makeMeasures()
                    p.removeByClass(stream.Measure)
                    for m in ms:
                        p.append(m)
        else:
            try:
                s.makeMeasures(inPlace=True)
            except Exception:
                pass
    except Exception:
        pass

    return has_measure_numbers(s)
""", language="python")

st.header("Create a demo multi-part score (melody + bass) with explicit measures")
st.code("""
def create_demo_score():
    # Melody part with explicit measures
    p1 = stream.Part()
    p1.insert(0, instrument.Piano())
    p1.append(meter.TimeSignature('4/4'))
    p1.append(tempo.MetronomeMark(number=90))

    m1 = stream.Measure(number=1)
    m1.append(note.Note('E4', quarterLength=1))
    m1.append(note.Note('D4', quarterLength=1))
    m1.append(note.Note('C4', quarterLength=2))
    p1.append(m1)

    m2 = stream.Measure(number=2)
    m2.append(note.Rest(quarterLength=1))
    m2.append(note.Note('G4', quarterLength=1))
    m2.append(note.Note('E4', quarterLength=2))
    p1.append(m2)

    # Bass part with explicit measures
    p2 = stream.Part()
    p2.insert(0, instrument.Piano())
    bm1 = stream.Measure(number=1)
    bm1.append(note.Note('C3', quarterLength=2))
    bm1.append(note.Note('F2', quarterLength=2))
    p2.append(bm1)
    bm2 = stream.Measure(number=2)
    bm2.append(note.Note('G2', quarterLength=2))
    bm2.append(note.Note('C3', quarterLength=2))
    p2.append(bm2)

    sc = stream.Score([p1, p2])
    return sc

score = create_demo_score()
print('Created demo score with', len(score.parts), 'parts')
print('Ensure measures present:', ensure_score_has_measures(score))
""", language="python")

st.header("Chordify the score and inspect chord events")
st.code("""
chordified = score.chordify()
print('Text view of chordified stream:')
chordified.show('text')
chords_list = list(chordified.recurse().getElementsByClass('Chord'))
print('Number of chord events:', len(chords_list))
""", language="python")

st.header("Infer a global key and extract chord-level DataFrame")
st.code("""
try:
    global_key = score.analyze('key')
    print('Inferred global key:', global_key)
except Exception as e:
    global_key = None
    print('Global key inference failed:', e)

rows = []
for ch in chords_list:
    pitches = [p.nameWithOctave for p in ch.pitches]
    root = None
    try:
        root = ch.root().name
    except Exception:
        root = None
    common_name = ch.commonName
    inv = ch.inversion()
    off = ch.offset
    dur = ch.duration.quarterLength
    measure_num = getattr(ch, 'measureNumber', None)
    try:
        measure_num = int(measure_num) if measure_num is not None else None
    except Exception:
        measure_num = None

    rn_fig = None
    if global_key is not None:
        try:
            rn = roman.romanNumeralFromChord(ch, global_key)
            rn_fig = rn.figure
        except Exception:
            rn_fig = None

    rows.append({'offset': off, 'duration': dur, 'pitches': ', '.join(pitches), 'root': root, 'common_name': common_name, 'inversion': inv, 'measure_num': measure_num, 'roman_numeral_global': rn_fig})

df_chords = pd.DataFrame(rows)
df_chords
""", language="python")

st.header("Measure-by-measure key inference")
st.markdown("We infer a key for each measure (the demo score includes measures so this will use them).")
st.code("""
from music21 import chord as m21chord

measure_keys = []
measure_numbers = sorted(df_chords['measure_num'].dropna().unique().astype(int))
for m in measure_numbers:
    group = df_chords[df_chords['measure_num'] == m]
    if group.empty:
        measure_keys.append({'measure': int(m), 'inferred_key': None})
        continue
    sub = stream.Stream()
    for _, r in group.iterrows():
        try:
            pcs = [pitch.Pitch(p) for p in r['pitches'].split(', ') if p.strip()]
            if pcs:
                temp_ch = m21chord.Chord(pcs)
                sub.append(temp_ch)
        except Exception:
            continue
    try:
        k = sub.analyze('key') if len(sub) > 0 else None
        measure_keys.append({'measure': int(m), 'inferred_key': k.name if k is not None else None})
    except Exception:
        measure_keys.append({'measure': int(m), 'inferred_key': None})

df_measure_keys = pd.DataFrame(measure_keys)
df_measure_keys
""", language="python")

st.header("Compute Roman numerals per chord using local measure keys (fallback to global)")
st.code("""
df_local = df_chords.merge(df_measure_keys, how='left', left_on='measure_num', right_on='measure')
local_roman = []
for _, r in df_local.iterrows():
    pitches_text = r['pitches']
    try:
        pcs = [pitch.Pitch(p) for p in pitches_text.split(', ') if p.strip()]
        if not pcs:
            local_roman.append(None)
            continue
        temp_ch = m21chord.Chord(pcs)
    except Exception:
        local_roman.append(None)
        continue

    local_key_name = r.get('inferred_key') if 'inferred_key' in r.index else None
    key_obj = None
    if local_key_name and pd.notna(local_key_name):
        try:
            key_obj = key.Key(local_key_name)
        except Exception:
            key_obj = None
    if key_obj is None and global_key is not None:
        key_obj = global_key

    if key_obj is None:
        local_roman.append(None)
        continue

    try:
        rn = roman.romanNumeralFromChord(temp_ch, key_obj)
        local_roman.append(rn.figure)
    except Exception:
        local_roman.append(None)

df_local['roman_numeral_local'] = local_roman
df_local[['offset','measure_num','pitches','inferred_key','roman_numeral_local']]
""", language="python")

st.header("Export chordified MIDI (and optionally render to WAV)")
st.code("""
midi_fp = 'chordified_demo.mid'
chordified.write('midi', fp=midi_fp)
print('Wrote MIDI to', midi_fp)

# Rendering (uncomment when FluidSynth + soundfont present)
sf2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
if os.path.exists(sf2):
     wav_fp = 'chordified_demo.wav'
     cmd = f"fluidsynth -ni '{sf2}' {midi_fp} -F {wav_fp} -r 44100"
     os.system(cmd)
     if os.path.exists(wav_fp):
         display(Audio(wav_fp))
     else:
         print('FluidSynth did not produce a WAV.')
""", language="python")

st.header("Notes")
st.markdown("""
- This notebook uses only the created demo score; there are no upload or parsing steps.
- The demo score includes explicit Measure objects (numbers 1 and 2) so measure-level analysis works reliably.
- If you want to test with a different demo score, edit the create_demo_score() function and re-run the notebook.
""")

st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")