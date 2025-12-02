import streamlit as st
from music21 import converter, note, chord
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tempfile
import os

# ── Page config (wide layout) ─────────────────────────────────────
st.set_page_config(layout="wide", page_title="MusicXML Pitch Visualizer")

st.title("MusicXML Pitch-Class Visualizer")
st.markdown("""
Upload one or two MusicXML files.  
Each visualization has its own **measure slider** and uses the **same color for the same pitch class**  
(e.g., every C — in any octave — is the same color).
""")

# ── Uploaders ─────────────────────────────────────────────────────
left_file = st.file_uploader("Left file (required)", type=["xml", "mxl"])
right_file = st.file_uploader("Right file (optional)", type=["xml", "mxl"])

# ── Palette selector ───────────────────────────────────────────────
palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'tab10', 'Set1', 'Set2', 'Set3', 'Pastel1']
palette = st.selectbox("Color palette", palettes, index=0)

# Fixed pitch-class names and colors (12 chromatic classes)
pitch_classes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
colors = sns.color_palette(palette, 12)
color_dict = {i: colors[i] for i in range(12)}   # 0-11 → fixed color

# ── Helper: convert uploaded file → DataFrame ─────────────────────
def file_to_df(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mxl') as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        stream = converter.parse(tmp_path)
        os.unlink(tmp_path)

        rows = []
        for el in stream.flat.getElementsByClass([note.Note, chord.Chord]):
            offset = el.offset
            dur = el.duration.quarterLength
            measure = getattr(el, 'measureNumber', None)

            if isinstance(el, note.Note):
                p = el.pitch
                rows.append({
                    'offset': offset,
                    'pitch': p.midi,
                    'duration': dur,
                    'pitch_class': p.name,
                    'pc_mod': p.pitchClass,      # 0-11
                    'measure': measure
                })
            else:  # chord
                for p in el.pitches:
                    rows.append({
                        'offset': offset,
                        'pitch': p.midi,
                        'duration': dur,
                        'pitch_class': p.name,
                        'pc_mod': p.pitchClass,
                        'measure': measure
                    })
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ── Load data only when a new file is uploaded (cached in session_state) ──
if left_file:
    if st.session_state.get("left_df") is None or st.session_state.get("left_name") != left_file.name:
        with st.spinner("Parsing left file…"):
            st.session_state.left_df = file_to_df(left_file)
            st.session_state.left_name = left_file.name

if right_file:
    if st.session_state.get("right_df") is None or st.session_state.get("right_name") != right_file.name:
        with st.spinner("Parsing right file…"):
            st.session_state.right_df = file_to_df(right_file)
            st.session_state.right_name = right_file.name

# ── Display visualizations ────────────────────────────────────────
left_df  = st.session_state.get("left_df")
right_df = st.session_state.get("right_df")

if left_df is None and right_df is None:
    st.info("Upload at least one MusicXML file to begin.")
else:
    # Layout: two columns if both exist, otherwise one wide column
    cols = st.columns(2) if left_df is not None and right_df is not None else [st]

    def plot_df(df, col, title):
        if df is None or df.empty:
            col.warning(f"No data for {title}")
            return

        # ── Measure slider ──
        measures = df['measure'].dropna().astype(int)
        if measures.empty:
            col.warning("No measure numbers – showing all data")
            filtered = df
            measure_range = (0, 0)
        else:
            min_m = int(measures.min())
            max_m = int(measures.max())
            measure_range = col.slider(
                f"Measures – {title}",
                min_value=min_m,
                max_value=max_m,
                value=(min_m, max_m),
                key=f"slider_{title}"
            )
            filtered = df[(df['measure'] >= measure_range[0]) & (df['measure'] <= measure_range[1])]

        # ── Plot ──
        fig, ax = plt.subplots(figsize=(11, 6))

        if filtered.empty:
            ax.text(0.5, 0.5, "No notes in selected range", ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1); ax.set_ylim(21, 109)
        else:
            sns.scatterplot(
                data=filtered,
                x='offset',
                y='pitch',
                hue='pc_mod',
                palette=color_dict,
                size='duration',
                sizes=(30, 250),
                ax=ax,
                legend=False          # we build our own legend
            )

        ax.set_title(f"{title} – Measures {measure_range[0]}–{measure_range[1]}")
        ax.set_xlabel("Time (quarter-note offsets)")
        ax.set_ylabel("MIDI Pitch")

        # ── Smart legend: only pitch classes that actually appear ──
        if not filtered.empty:
            present_pc = sorted(filtered['pc_mod'].unique())
            present_labels = [pitch_classes[i] for i in present_pc]

            handles = [
                Line2D([0], [0], marker='o', color='w',
                       label=label,
                       markerfacecolor=color_dict[pc],
                       markersize=10)
                for pc, label in zip(present_pc, present_labels)
            ]

            ax.legend(handles=handles,
                      title="Pitch Class",
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')

        col.pyplot(fig)

        with col.expander("Data table"):
            col.dataframe(filtered)

    # Plot the files
    if left_df is not None:
        plot_df(left_df, cols[0], "Left File")
    if right_df is not None:
        plot_df(right_df, cols[1] if len(cols) > 1 else cols[0], "Right File")