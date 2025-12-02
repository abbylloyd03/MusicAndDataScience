import streamlit as st
from music21 import converter, note, chord, pitch, stream
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
import tempfile
import os

# ── Page config (wide layout) ─────────────────────────────────────
st.set_page_config(layout="wide", page_title="MusicXML Pitch Visualizer")

st.title("MusicXML Pitch-Class Visualizer")
st.markdown(
    """
    Upload one or two MusicXML files (`.xml` or `.mxl`).

    - Use the **measure slider** for each file to select the range to visualize. The app infers the key from the selected range when `Color by -> Pitch relative to key` is chosen.
    - Choose a color palette to map pitch classes (or scale degrees) to colors. Select `Custom` to pick a specific color for each pitch class or scale degree.
    - `Color by` controls whether notes are colored by absolute pitch class (C, C#/Db, ...) or by scale degree relative to the inferred key (1 = tonic).
    - Expand the **Data table** under a plot to inspect the note offsets, MIDI pitches, durations, and inferred file key.

    Tip: If the selected range is too small to yield a key, the app will fall back to the file-wide key (if available) and will show a short notice.
    """
)

# ── Uploaders ─────────────────────────────────────────────────────
# discover samples folder and available sample files so we can show
# a sample-select option next to each uploader
samples_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'samples'))
try:
    os.makedirs(samples_dir, exist_ok=True)
except Exception:
    pass

sample_files = []
try:
    for fn in os.listdir(samples_dir):
        if fn.lower().endswith(('.xml', '.mxl')):
            sample_files.append(fn)
except Exception:
    sample_files = []

# Global input-mode toggle using Streamlit's toggle widget when available.
# Falls back to a checkbox for older Streamlit versions.
if hasattr(st, 'toggle'):
    use_samples = st.toggle("Use sample files for both panes", value=False, key='use_samples_both')
else:
    use_samples = st.checkbox("Use sample files for both panes", value=False, key='use_samples_both')

# Map boolean to the previous `mode` values used by the UI code
mode = "Choose sample" if use_samples else "Upload file"

# If the user just switched from Upload -> Choose sample, clear any uploaded-file
# visualizations so the panes are empty until a sample is selected.
prev_use = st.session_state.get('prev_use_samples_both')
if prev_use is None:
    # initialize previous state tracking
    st.session_state['prev_use_samples_both'] = use_samples
else:
    # switched from False -> True
    if prev_use is False and use_samples is True:
        # Clear cached DataFrames/names so previous uploaded plots disappear.
        if st.session_state.get('left_df') is not None:
            st.session_state.left_df = None
            st.session_state.left_name = None
        if st.session_state.get('right_df') is not None:
            st.session_state.right_df = None
            st.session_state.right_name = None
    # update previous-state for next run
    st.session_state['prev_use_samples_both'] = use_samples

# Show uploaders/selectors side-by-side
left_col, right_col = st.columns(2)

# Left side UI follows the global `mode`
left_file = None
left_choice = None
if mode == "Upload file":
    left_file = left_col.file_uploader("File 1 (required)", type=["xml", "mxl"], key='uploader_left')
    # If a file upload occurs, cancel any pending sample load for left
    if left_file is not None and st.session_state.get('sample_left_to_load'):
        st.session_state.pop('sample_left_to_load', None)
else:
    if sample_files:
        left_choice = left_col.selectbox("Choose sample for File 1", ["(none)"] + sample_files, key='sample_left_choice_ui')
        if left_choice and left_choice != "(none)":
            st.session_state['sample_left_to_load'] = left_choice
            st.session_state['ignore_uploader_left'] = True
    else:
        left_col.info("No sample files available")

# Right side UI follows the same global `mode`
right_file = None
right_choice = None
if mode == "Upload file":
    right_file = right_col.file_uploader("File 2 (optional)", type=["xml", "mxl"], key='uploader_right')
    if right_file is not None and st.session_state.get('sample_right_to_load'):
        st.session_state.pop('sample_right_to_load', None)
else:
    if sample_files:
        right_choice = right_col.selectbox("Choose sample for File 2", ["(none)"] + sample_files, key='sample_right_choice_ui')
        if right_choice and right_choice != "(none)":
            st.session_state['sample_right_to_load'] = right_choice
            st.session_state['ignore_uploader_right'] = True
    else:
        right_col.info("No sample files available")

# If the user cleared an uploaded file while in Upload mode, remove its cached DataFrame
# (but don't remove sample-loaded data when in Choose-sample mode).
if mode == "Upload file" and left_file is None and st.session_state.get("left_df") is not None:
    st.session_state.left_df = None
    st.session_state.left_name = None

if mode == "Upload file" and right_file is None and st.session_state.get("right_df") is not None:
    st.session_state.right_df = None
    st.session_state.right_name = None

# ── Palette selector ───────────────────────────────────────────────
palettes = ['Custom', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'tab10', 'Set1', 'Set2', 'Set3', 'Pastel1']
palette = st.selectbox("Color palette", palettes, index=0)

# Color-by selector: allow coloring by pitch class (pc_mod) or by pitch relative to key
color_by = st.selectbox("Color by", ["Pitch class", "Pitch relative to key"], index=0)

# Fixed pitch-class names (12 chromatic classes)
pitch_classes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

# Build color mappings. If the user selects the special 'Custom' palette,
# show 12 color pickers (pitch classes or scale degrees depending on
# `color_by`) and build the mapping from the chosen hex values. Otherwise,
# sample 12 colors from the chosen seaborn/matplotlib palette.
if palette != 'Custom':
    colors = sns.color_palette(palette, 12)
    color_dict_pc = {i: colors[i] for i in range(12)}   # 0-11 → fixed color for pitch classes
    # for pitch relative to key, values are 1-12 -> map 1->colors[0], 2->colors[1], ...
    color_dict_rel = {i+1: colors[i] for i in range(12)}
else:
    # Provide sensible defaults sampled from a standard palette
    default_colors = sns.color_palette('tab10', 12)

    # Helper to ensure values are hex strings for color_picker defaults
    def _hex_default(idx):
        try:
            return to_hex(default_colors[idx])
        except Exception:
            return '#777777'

    # Show color pickers in a compact grid. Keys are stable so values persist.
    if color_by == 'Pitch class':
        st.markdown("**Custom colors — pitch classes**")
        cols_cp = st.columns(4)
        custom_colors_pc = []
        for i, pc in enumerate(pitch_classes):
            col = cols_cp[i % 4]
            key = f'custom_color_pc_{i}'
            default = st.session_state.get(key, _hex_default(i))
            c = col.color_picker(f"{pc}", value=default, key=key)
            custom_colors_pc.append(c)
        color_dict_pc = {i: custom_colors_pc[i] for i in range(12)}
        # also build rel mapping so plotting code can always reference it
        color_dict_rel = {i+1: custom_colors_pc[i] for i in range(12)}
    else:
        st.markdown("**Custom colors — scale degrees (1 = tonic)**")
        cols_rel = st.columns(4)
        custom_colors_rel = []
        for i in range(12):
            degree = i+1
            col = cols_rel[i % 4]
            label = f"{degree}{' (tonic)' if degree==1 else ''}"
            key = f'custom_color_rel_{degree}'
            default = st.session_state.get(key, _hex_default(i))
            c = col.color_picker(label, value=default, key=key)
            custom_colors_rel.append(c)
        color_dict_rel = {i+1: custom_colors_rel[i] for i in range(12)}
        # also build pc mapping for completeness
        color_dict_pc = {i: custom_colors_rel[i] for i in range(12)}

# ── Helper: convert uploaded file → DataFrame ─────────────────────
def file_to_df(uploaded_file):
    """Parse uploaded MusicXML and return (DataFrame, stream).

    The DataFrame contains note-level rows with pitch, pc_mod, measure, offset, duration.
    We no longer compute pitch-relative-to-key per-measure here; that is inferred
    dynamically from the currently selected measure range when plotting.
    """
    if uploaded_file is None:
        return None, None
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mxl') as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        stream = converter.parse(tmp_path)
        os.unlink(tmp_path)

        # Infer the overall key of the piece using music21's analysis (kept for reference)
        try:
            key_obj = stream.analyze('key')
            file_key = key_obj.name
        except Exception:
            file_key = None

        rows = []
        for el in stream.flatten().getElementsByClass([note.Note, chord.Chord]):
            offset = el.offset
            dur = el.duration.quarterLength
            measure = getattr(el, 'measureNumber', None)

            if isinstance(el, note.Note):
                p = el.pitch
                # try to coerce measure to an int for plotting; keep original
                measure_num = None
                try:
                    if measure is not None:
                        measure_num = int(measure)
                except Exception:
                    measure_num = None
                rows.append({
                    'offset': offset,
                    'pitch': p.midi,
                    'duration': dur,
                    'pitch_class': p.name,
                    'pc_mod': p.pitchClass,      # 0-11
                    'measure': measure,
                    'measure_num': measure_num,
                    'file_key': file_key,
                })
            else:  # chord
                for p in el.pitches:
                    measure_num = None
                    try:
                        if measure is not None:
                            measure_num = int(measure)
                    except Exception:
                        measure_num = None
                    rows.append({
                        'offset': offset,
                        'pitch': p.midi,
                        'duration': dur,
                        'pitch_class': p.name,
                        'pc_mod': p.pitchClass,
                        'measure': measure,
                        'measure_num': measure_num,
                        'file_key': file_key,
                    })
        df = pd.DataFrame(rows) if rows else None
        # Convert measure_num to a nullable integer dtype for safe numeric operations.
        # Some MusicXML files contain non-standard measure objects; convert
        # each value individually to avoid pandas' internal errors.
        if df is not None and 'measure_num' in df.columns:
            def _safe_int(v):
                try:
                    if v is None:
                        return pd.NA
                    if v is pd.NA:
                        return pd.NA
                    # avoid treating booleans as ints
                    if isinstance(v, bool):
                        return pd.NA
                    # numeric ints are fine
                    if isinstance(v, int):
                        return int(v)
                    # numpy integer types
                    try:
                        import numpy as _np
                        if isinstance(v, (_np.integer,)):
                            return int(v)
                    except Exception:
                        pass
                    # fall back to converting from string representation
                    s = str(v).strip()
                    if s == '':
                        return pd.NA
                    return int(s)
                except Exception:
                    return pd.NA

            try:
                df['measure_num'] = df['measure_num'].apply(_safe_int).astype('Int64')
            except Exception:
                # Extreme fallback: coerce everything to NA
                df['measure_num'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')
        return df, stream
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

# If the uploader-area Load buttons placed a sample to load into session_state,
# handle the actual parsing here (file_to_df is defined at this point).
if st.session_state.get('sample_left_to_load'):
    chosen = st.session_state.pop('sample_left_to_load')
    try:
        path = os.path.join(samples_dir, chosen)
        with open(path, 'rb') as fh:
            data = fh.read()
        import io
        bio = io.BytesIO(data)
        bio.name = chosen
        df_s, s_obj = file_to_df(bio)
        st.session_state.left_df = df_s
        st.session_state.left_stream = s_obj
        st.session_state.left_name = f"sample:{chosen}"
        st.success(f"Loaded sample {chosen} into File 1")
    except Exception as e:
        st.error(f"Could not load sample: {e}")

if st.session_state.get('sample_right_to_load'):
    chosen = st.session_state.pop('sample_right_to_load')
    try:
        path = os.path.join(samples_dir, chosen)
        with open(path, 'rb') as fh:
            data = fh.read()
        import io
        bio = io.BytesIO(data)
        bio.name = chosen
        df_s, s_obj = file_to_df(bio)
        st.session_state.right_df = df_s
        st.session_state.right_stream = s_obj
        st.session_state.right_name = f"sample:{chosen}"
        st.success(f"Loaded sample {chosen} into File 2")
    except Exception as e:
        st.error(f"Could not load sample: {e}")

# ── Load data only when a new file is uploaded (cached in session_state) ──
# If a sample selection has been made that should override uploaders, skip uploader parsing.
if st.session_state.pop('ignore_uploader_left', False):
    # uploader intentionally ignored because user selected a sample
    pass
elif left_file:
    if st.session_state.get("left_df") is None or st.session_state.get("left_name") != getattr(left_file, 'name', None):
        with st.spinner("Parsing left file…"):
            df, stream = file_to_df(left_file)
            st.session_state.left_df = df
            st.session_state.left_stream = stream
            st.session_state.left_name = left_file.name

if st.session_state.pop('ignore_uploader_right', False):
    pass
elif right_file:
    if st.session_state.get("right_df") is None or st.session_state.get("right_name") != getattr(right_file, 'name', None):
        with st.spinner("Parsing right file…"):
            df, stream = file_to_df(right_file)
            st.session_state.right_df = df
            st.session_state.right_stream = stream
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
        # Use the numeric `measure_num` column (safely converted at parse time)
        measures = df.get('measure_num')
        if measures is None or measures.dropna().empty:
            col.warning("No measure numbers – showing all data")
            filtered = df
            measure_range = (0, 0)
        else:
            min_m = int(measures.dropna().min())
            max_m = int(measures.dropna().max())
            measure_range = col.slider(
                f"Measures – {title}",
                min_value=min_m,
                max_value=max_m,
                value=(min_m, max_m),
                key=f"slider_{title}"
            )
            # filter using the numeric measure column
            filtered = df[(df['measure_num'] >= measure_range[0]) & (df['measure_num'] <= measure_range[1])]

        # ── Plot ──
        fig, ax = plt.subplots(figsize=(11, 6))

        if filtered.empty:
            ax.text(0.5, 0.5, "No notes in selected range", ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1); ax.set_ylim(21, 109)
        else:
            # choose color field and palette mapping based on user selection
            if color_by == "Pitch class":
                color_field = 'pc_mod'
                palette_map = color_dict_pc
            else:
                # For pitch-relative coloring we infer the key from the
                # currently selected measure range and add a temporary
                # `pitch_rel_to_key` column to `filtered`.
                color_field = 'pitch_rel_to_key'
                palette_map = color_dict_rel

                # Infer key from selected measures using the parsed stream
                # stored in session_state (left_stream / right_stream).
                stream_key = None
                if title == 'File 1':
                    stream_key = st.session_state.get('left_stream')
                else:
                    stream_key = st.session_state.get('right_stream')

                def tonic_pc_from_key_name(key_name):
                    if not key_name:
                        return None
                    try:
                        tonic_name = str(key_name).split()[0]
                        return pitch.Pitch(tonic_name).pitchClass
                    except Exception:
                        return None

                tonic_pc = None
                inferred_key_name = None
                if stream_key is not None:
                    try:
                        # collect elements within the selected measure range (coerce measure numbers safely)
                        elems = []
                        for el in stream_key.flatten().getElementsByClass([note.Note, chord.Chord]):
                            mnum = getattr(el, 'measureNumber', None)
                            if mnum is None:
                                continue
                            try:
                                mnum_i = int(mnum)
                            except Exception:
                                continue
                            if measure_range[0] <= mnum_i <= measure_range[1]:
                                elems.append(el)

                        if elems:
                            sub = stream.Stream()
                            for el in elems:
                                sub.insert(el.offset, el)
                            try:
                                k = sub.analyze('key')
                                inferred_key_name = k.name
                                tonic_pc = tonic_pc_from_key_name(k.name)
                            except Exception:
                                tonic_pc = None
                    except Exception:
                        tonic_pc = None
                # If selected-range analysis failed, fall back to the file-wide key
                # (stored in the `file_key` column). Also show a small info message
                # so the user knows which key was used.
                if tonic_pc is None:
                    # try file-wide key from the filtered rows
                    try:
                        file_keys = filtered['file_key'].dropna().unique()
                        if len(file_keys) > 0:
                            fk = file_keys[0]
                            tonic_pc = tonic_pc_from_key_name(fk)
                            if tonic_pc is not None:
                                inferred_key_name = fk
                    except Exception:
                        tonic_pc = None

                filtered = filtered.copy()
                if tonic_pc is not None:
                    filtered['pitch_rel_to_key'] = ((filtered['pc_mod'].astype(int) - tonic_pc) % 12) + 1
                    if inferred_key_name:
                        # If the file-wide key differs from the inferred key for the
                        # selected range, show both so the user understands the difference.
                        try:
                            file_keys = filtered['file_key'].dropna().unique()
                            file_key_name = file_keys[0] if len(file_keys) > 0 else None
                        except Exception:
                            file_key_name = None

                        if file_key_name and file_key_name != inferred_key_name:
                            col.info(
                                f"Coloring by scale degree inferred from: {inferred_key_name} (selected range). File-wide key: {file_key_name}."
                            )
                        else:
                            col.info(f"Coloring by scale degree inferred from: {inferred_key_name}")
                else:
                    filtered['pitch_rel_to_key'] = None
                    col.info("Could not infer a key for the selected range; no relative coloring applied.")

            # Prepare a sanitized copy for plotting to avoid seaborn/pandas
            # errors when values contain unexpected object types.
            plotting_df = filtered.copy()
            # Coerce numeric plotting columns; invalid values become NaN
            for _col in ('offset', 'pitch', 'duration'):
                if _col in plotting_df.columns:
                    plotting_df[_col] = pd.to_numeric(plotting_df[_col], errors='coerce')

            # Coerce hue column to numeric when expected (pc_mod, pitch_rel_to_key)
            if color_field in ('pc_mod', 'pitch_rel_to_key') and color_field in plotting_df.columns:
                plotting_df[color_field] = pd.to_numeric(plotting_df[color_field], errors='coerce')
            else:
                # ensure hue is string-like so seaborn treats it categorically
                if color_field in plotting_df.columns:
                    plotting_df[color_field] = plotting_df[color_field].astype('str')

            # Drop rows that lack essential numeric coordinates
            plotting_df = plotting_df.dropna(subset=['offset', 'pitch'])

            # Ensure duration has a fallback numeric value
            if 'duration' in plotting_df.columns:
                plotting_df['duration'] = plotting_df['duration'].fillna(plotting_df['duration'].median() if not plotting_df['duration'].dropna().empty else 1.0)

            sns.scatterplot(
                data=plotting_df,
                x='offset',
                y='pitch',
                hue=color_field if color_field in plotting_df.columns else None,
                palette=palette_map,
                size='duration' if 'duration' in plotting_df.columns else None,
                sizes=(30, 250),
                ax=ax,
                legend=False          # we build our own legend
            )

            # Add measure-number labels on the x-axis: place a tick roughly
            # at the mean offset for each measure and label it with the measure
            # number. If there are many measures, reduce label density.
            try:
                measures_present = filtered.get('measure_num')
                if measures_present is not None and not measures_present.dropna().empty:
                    ticks = []
                    for m in sorted(measures_present.dropna().astype(int).unique()):
                        subset = filtered[filtered['measure_num'] == m]
                        if not subset.empty:
                            tick_pos = float(subset['offset'].mean())
                            ticks.append((tick_pos, m))

                    # Avoid overcrowding x-axis labels
                    max_labels = 20
                    if len(ticks) > max_labels:
                        step = max(1, len(ticks) // max_labels)
                        ticks = [t for i, t in enumerate(ticks) if i % step == 0]

                    tick_positions = [t for t, _ in ticks]
                    tick_labels = [str(m) for _, m in ticks]
                    if tick_positions:
                        ax.set_xticks(tick_positions)
                        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
            except Exception:
                # If anything goes wrong here, fall back to default x-axis.
                pass

        ax.set_title(f"{title} – Measures {measure_range[0]}–{measure_range[1]}")
        ax.set_xlabel("Measure")
        ax.set_ylabel("MIDI Pitch")

        # ── Smart legend: only pitch classes that actually appear ──
        if not filtered.empty:
            # Build legend entries based on current color selection
            if color_by == "Pitch class":
                present_vals = sorted(filtered['pc_mod'].dropna().astype(int).unique())
                present_labels = [pitch_classes[i] for i in present_vals]
                palette_map = color_dict_pc
            else:
                present_vals = sorted(filtered['pitch_rel_to_key'].dropna().astype(int).unique())
                # label scale degrees; mark tonic (1)
                present_labels = [f"{i}{' (tonic)' if i==1 else ''}" for i in present_vals]
                palette_map = color_dict_rel

            handles = [
                Line2D([0], [0], marker='o', color='w',
                       label=label,
                       markerfacecolor=palette_map[val],
                       markersize=10)
                for val, label in zip(present_vals, present_labels)
            ]

            legend_title = "Pitch Class" if color_by == "Pitch class" else "Scale Degree"
            ax.legend(handles=handles,
                      title=legend_title,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')

        col.pyplot(fig)

        exp = col.expander("Data table")
        # Ensure the table shows columns in a consistent, useful order.
        ordered_cols = [
            'measure', 'offset', 'duration', 'pitch', 'pitch_class',
            'pc_mod', 'file_key', 'pitch_rel_to_key'
        ]
        # Make a safe copy and add any missing columns (filled with None)
        table_df = filtered.copy()
        for c in ordered_cols:
            if c not in table_df.columns:
                table_df[c] = None

        exp.dataframe(table_df[ordered_cols])

    # Plot the files
    if left_df is not None:
        plot_df(left_df, cols[0], "File 1")
    if right_df is not None:
        plot_df(right_df, cols[1] if len(cols) > 1 else cols[0], "File 2")