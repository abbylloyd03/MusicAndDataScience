"""
Supervised ML Application for Clarinet Onset Detection

This application trains on WAV audio files and corresponding SVL annotation files
(from Sonic Visualiser) that contain articulation onsets and sustain onsets.
The trained model can then identify these onset types in new clarinet recordings.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import xml.etree.ElementTree as ET
from io import BytesIO

# Audio processing and ML imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Onset Detection ML")

st.title("Clarinet Onset Detection - Supervised ML Application")

st.markdown("""
This application uses supervised machine learning to identify **articulation onsets** 
and **sustain onsets** in solo clarinet recordings.

### How it works:
1. **Upload Training Data**: Provide WAV audio files with their corresponding SVL 
   annotation files (from Sonic Visualiser)
2. **Train Model**: The app extracts audio features and trains a classifier
3. **Predict**: Upload new recordings to detect onset types

### Annotation Types:
- **Articulation (Attack) Onsets**: The point when an articulation begins
- **Sustain Onsets**: The point when the articulation has settled into sustained sound
""")

# Check dependencies
if not LIBROSA_AVAILABLE:
    st.error("librosa is not installed. Please install it with: `pip install librosa`")
    st.stop()

if not SKLEARN_AVAILABLE:
    st.error("scikit-learn is not installed. Please install it with: `pip install scikit-learn`")
    st.stop()


# ── SVL Parser ─────────────────────────────────────────────────────
def parse_svl_file(svl_content):
    """
    Parse a Sonic Visualiser SVL file and extract frame positions.
    
    Args:
        svl_content: String or bytes content of the SVL file
        
    Returns:
        dict with 'frames' (list of frame numbers), 'sample_rate' (int)
    """
    if isinstance(svl_content, bytes):
        svl_content = svl_content.decode('utf-8')
    
    root = ET.fromstring(svl_content)
    
    # Find the model element to get sample rate
    model = root.find('.//model')
    sample_rate = 44100  # default
    if model is not None:
        sample_rate = int(model.get('sampleRate', 44100))
    
    # Find all point elements and extract frames
    frames = []
    for point in root.findall('.//point'):
        frame = point.get('frame')
        if frame is not None:
            frames.append(int(frame))
    
    return {
        'frames': sorted(frames),
        'sample_rate': sample_rate
    }


def frames_to_times(frames, sample_rate):
    """Convert frame numbers to time in seconds."""
    return [frame / sample_rate for frame in frames]


def save_content_to_file(file_path, content):
    """
    Save content to a file, handling both bytes and string content.
    
    Args:
        file_path: Path to save the file
        content: Content to write (bytes or string)
    """
    with open(file_path, 'wb') as f:
        if isinstance(content, bytes):
            f.write(content)
        else:
            f.write(content.encode('utf-8'))


# ── Feature Extraction ─────────────────────────────────────────────
def extract_features_at_onset(y, sr, onset_time, window_size=0.05):
    """
    Extract audio features around an onset time.
    
    Args:
        y: Audio signal
        sr: Sample rate
        onset_time: Time in seconds
        window_size: Window size in seconds around the onset
        
    Returns:
        dict of features
    """
    # Convert time to sample index
    center_sample = int(onset_time * sr)
    window_samples = int(window_size * sr)
    
    start_sample = max(0, center_sample - window_samples // 2)
    end_sample = min(len(y), center_sample + window_samples // 2)
    
    if end_sample - start_sample < window_samples // 4:
        return None
    
    segment = y[start_sample:end_sample]
    
    if len(segment) < 512:
        # Pad if too short
        segment = np.pad(segment, (0, 512 - len(segment)), mode='constant')
    
    features = {}
    
    # Time-domain features
    features['rms'] = float(np.sqrt(np.mean(segment**2)))
    features['zero_crossing_rate'] = float(np.sum(np.abs(np.diff(np.sign(segment)))) / (2 * len(segment)))
    
    # Attack characteristics
    envelope = np.abs(segment)
    if len(envelope) > 1:
        # Attack slope (how quickly amplitude rises)
        max_idx = np.argmax(envelope)
        if max_idx > 0:
            features['attack_slope'] = float(envelope[max_idx] / (max_idx / sr))
        else:
            features['attack_slope'] = 0.0
        
        # Peak to mean ratio
        features['peak_to_mean'] = float(np.max(envelope) / (np.mean(envelope) + 1e-10))
    else:
        features['attack_slope'] = 0.0
        features['peak_to_mean'] = 1.0
    
    # Spectral features
    n_fft = min(len(segment), 2048)
    if n_fft >= 512:
        stft = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=n_fft//4))
        
        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
        features['spectral_centroid_std'] = float(np.std(spec_centroid))
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(S=stft, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spec_bw))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        
        # Spectral flatness (tonality)
        flatness = librosa.feature.spectral_flatness(S=stft)
        features['spectral_flatness_mean'] = float(np.mean(flatness))
        
        # Spectral flux (change in spectrum)
        if stft.shape[1] > 1:
            flux = np.mean(np.diff(stft, axis=1)**2)
            features['spectral_flux'] = float(flux)
        else:
            features['spectral_flux'] = 0.0
    else:
        features['spectral_centroid_mean'] = 0.0
        features['spectral_centroid_std'] = 0.0
        features['spectral_bandwidth_mean'] = 0.0
        features['spectral_rolloff_mean'] = 0.0
        features['spectral_flatness_mean'] = 0.0
        features['spectral_flux'] = 0.0
    
    # MFCCs (mel-frequency cepstral coefficients)
    try:
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=min(len(segment), 2048))
        for i in range(min(13, mfccs.shape[0])):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
    except Exception:
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
    
    return features


def extract_all_features(y, sr, onset_times, labels, window_size=0.05):
    """
    Extract features for all onsets.
    
    Args:
        y: Audio signal
        sr: Sample rate
        onset_times: List of onset times in seconds
        labels: List of labels (0 for attack, 1 for sustain)
        window_size: Window size in seconds around each onset
        
    Returns:
        DataFrame with features and labels
    """
    all_features = []
    valid_labels = []
    
    for onset_time, label in zip(onset_times, labels):
        features = extract_features_at_onset(y, sr, onset_time, window_size=window_size)
        if features is not None:
            features['onset_time'] = onset_time
            features['label'] = label
            all_features.append(features)
            valid_labels.append(label)
    
    if all_features:
        return pd.DataFrame(all_features)
    return pd.DataFrame()


# ── Model Training ─────────────────────────────────────────────────
def create_model(model_type, model_params):
    """
    Create a classifier based on model type and parameters.
    
    Args:
        model_type: Type of model ('Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network')
        model_params: Dictionary of model parameters
        
    Returns:
        Configured classifier
    """
    if model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 2),
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 3),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_type == "SVM":
        return SVC(
            C=model_params.get('C', 1.0),
            kernel=model_params.get('kernel', 'rbf'),
            gamma=model_params.get('gamma', 'scale'),
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    elif model_type == "Neural Network":
        return MLPClassifier(
            hidden_layer_sizes=model_params.get('hidden_layer_sizes', (100, 50)),
            activation=model_params.get('activation', 'relu'),
            learning_rate_init=model_params.get('learning_rate', 0.001),
            max_iter=model_params.get('max_iter', 500),
            random_state=42
        )
    else:
        # Default to Random Forest
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )


def train_model(features_df, model_type="Random Forest", model_params=None):
    """
    Train a classifier on the extracted features.
    
    Args:
        features_df: DataFrame with features and 'label' column
        model_type: Type of model to train
        model_params: Dictionary of model hyperparameters
        
    Returns:
        tuple (model, scaler, feature_columns, metrics_dict)
    """
    if model_params is None:
        model_params = {}
    
    # Separate features and labels
    feature_cols = [col for col in features_df.columns if col not in ['label', 'onset_time']]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = create_model(model_type, model_params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': float(np.mean(y_pred == y_test)),
        'classification_report': classification_report(y_test, y_pred, 
                                                        target_names=['Attack', 'Sustain'],
                                                        output_dict=True,
                                                        zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'model_type': model_type
    }
    
    return model, scaler, feature_cols, metrics


def predict_onsets(y, sr, model, scaler, feature_cols, threshold=0.1, window_size=0.05):
    """
    Detect and classify onsets in audio.
    
    Args:
        y: Audio signal
        sr: Sample rate
        model: Trained classifier
        scaler: Feature scaler
        feature_cols: List of feature column names
        threshold: Onset detection threshold
        window_size: Window size in seconds around each onset
        
    Returns:
        DataFrame with predicted onsets and their classifications
    """
    # Detect onsets using librosa
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    if len(onset_times) == 0:
        return pd.DataFrame()
    
    # Extract features for each detected onset
    all_features = []
    valid_times = []
    
    for onset_time in onset_times:
        features = extract_features_at_onset(y, sr, onset_time, window_size=window_size)
        if features is not None:
            all_features.append(features)
            valid_times.append(onset_time)
    
    if not all_features:
        return pd.DataFrame()
    
    # Create feature matrix
    features_df = pd.DataFrame(all_features)
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    results = pd.DataFrame({
        'onset_time': valid_times,
        'predicted_class': ['Attack' if p == 0 else 'Sustain' for p in predictions],
        'confidence': [max(prob) for prob in probabilities]
    })
    
    return results


# ── Visualization ─────────────────────────────────────────────────
def plot_waveform_with_onsets(y, sr, attack_times, sustain_times, title="Audio with Onsets"):
    """Plot waveform with marked onsets."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    times = np.arange(len(y)) / sr
    ax.plot(times, y, alpha=0.7, linewidth=0.5)
    
    # Mark attacks (red)
    for t in attack_times:
        ax.axvline(x=t, color='red', linestyle='--', alpha=0.7, label='Attack' if t == attack_times[0] else '')
    
    # Mark sustains (blue)
    for t in sustain_times:
        ax.axvline(x=t, color='blue', linestyle=':', alpha=0.7, label='Sustain' if t == sustain_times[0] else '')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    
    # Create legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols):
    """Plot feature importance from the trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Most Important Features')
    plt.tight_layout()
    return fig


# ── Articulation Timing Analysis ───────────────────────────────────
def pair_attack_sustain_onsets(attack_times, sustain_times):
    """
    Pair each attack onset with its corresponding sustain onset.
    
    Each attack is paired with the next sustain that occurs after it
    (before the next attack).
    
    Args:
        attack_times: List of attack onset times in seconds
        sustain_times: List of sustain onset times in seconds
        
    Returns:
        List of tuples: (attack_time, sustain_time, duration)
    """
    attack_times = sorted(attack_times)
    sustain_times = sorted(sustain_times)
    
    pairs = []
    sustain_idx = 0
    
    for i, attack_time in enumerate(attack_times):
        # Find the next attack time (if any) to set upper bound
        next_attack = attack_times[i + 1] if i + 1 < len(attack_times) else float('inf')
        
        # Find sustain that comes after this attack but before next attack
        while sustain_idx < len(sustain_times):
            sustain_time = sustain_times[sustain_idx]
            
            if sustain_time > attack_time and sustain_time < next_attack:
                # Valid pair found
                duration = sustain_time - attack_time
                pairs.append({
                    'attack_time': attack_time,
                    'sustain_time': sustain_time,
                    'attack_duration_ms': duration * 1000,  # Convert to milliseconds
                    'note_index': i + 1
                })
                sustain_idx += 1
                break
            elif sustain_time >= next_attack:
                # This sustain belongs to a later attack
                break
            else:
                # Sustain is before attack, skip it
                sustain_idx += 1
    
    return pairs


def analyze_timing_consistency(pairs_df):
    """
    Analyze the consistency of attack durations.
    
    Args:
        pairs_df: DataFrame with attack_duration_ms column
        
    Returns:
        dict with statistics
    """
    durations = pairs_df['attack_duration_ms'].values
    
    mean_val = float(np.mean(durations))
    std_val = float(np.std(durations))
    
    stats = {
        'count': len(durations),
        'mean_ms': mean_val,
        'std_ms': std_val,
        'min_ms': float(np.min(durations)),
        'max_ms': float(np.max(durations)),
        'median_ms': float(np.median(durations)),
        'cv_percent': float(std_val / mean_val * 100) if mean_val > 0 else float('nan'),
        'range_ms': float(np.max(durations) - np.min(durations))
    }
    
    # Identify outliers (beyond 2 standard deviations)
    pairs_df = pairs_df.copy()
    pairs_df['is_outlier'] = (pairs_df['attack_duration_ms'] < mean_val - 2*std_val) | (pairs_df['attack_duration_ms'] > mean_val + 2*std_val)
    stats['outlier_count'] = int(pairs_df['is_outlier'].sum())
    
    return stats, pairs_df


def plot_timing_histogram(pairs_df, stats):
    """Plot histogram of attack durations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    durations = pairs_df['attack_duration_ms'].values
    
    ax.hist(durations, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(stats['mean_ms'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_ms']:.1f} ms")
    ax.axvline(stats['median_ms'], color='green', linestyle=':', linewidth=2, label=f"Median: {stats['median_ms']:.1f} ms")
    
    ax.set_xlabel('Attack Duration (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Attack-to-Sustain Durations')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_timing_over_time(pairs_df, stats):
    """Plot attack durations over time to see consistency."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ['red' if outlier else 'steelblue' for outlier in pairs_df['is_outlier']]
    
    ax.scatter(pairs_df['attack_time'], pairs_df['attack_duration_ms'], 
               c=colors, s=50, alpha=0.7)
    
    # Add mean line
    ax.axhline(stats['mean_ms'], color='green', linestyle='--', linewidth=2, 
               label=f"Mean: {stats['mean_ms']:.1f} ms")
    
    # Add standard deviation band
    ax.fill_between([pairs_df['attack_time'].min(), pairs_df['attack_time'].max()],
                    stats['mean_ms'] - stats['std_ms'],
                    stats['mean_ms'] + stats['std_ms'],
                    alpha=0.2, color='green', label=f"±1 Std Dev: {stats['std_ms']:.1f} ms")
    
    ax.set_xlabel('Time in Recording (s)')
    ax.set_ylabel('Attack Duration (ms)')
    ax.set_title('Attack Duration Consistency Over Time')
    ax.legend()
    
    # Mark outliers in legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outlier (>2σ)'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_ms']:.1f} ms"),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_timing_boxplot(pairs_df):
    """Create a boxplot of attack durations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.boxplot(pairs_df['attack_duration_ms'].values, vert=True)
    ax.set_ylabel('Attack Duration (ms)')
    ax.set_title('Attack Duration Distribution')
    ax.set_xticklabels(['All Notes'])
    
    plt.tight_layout()
    return fig


# ── Session State Initialization ───────────────────────────────────
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.scaler = None
    st.session_state.feature_cols = None
    st.session_state.training_metrics = None
    st.session_state.window_size = 0.05


# ── Main Application Tabs ──────────────────────────────────────────
tab_train, tab_predict, tab_analyze, tab_about = st.tabs(["🎓 Train Model", "🎯 Predict", "📊 Analyze Timing", "ℹ️ About"])

with tab_train:
    st.header("Train Onset Detection Model")
    
    st.markdown("""
    Upload your training data:
    - **WAV file**: The audio recording
    - **Attacks SVL file**: Sonic Visualiser annotations for articulation/attack onsets
    - **Sustain SVL file**: Sonic Visualiser annotations for sustain onsets
    """)
    
    # File uploaders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wav_files = st.file_uploader(
            "WAV Audio Files",
            type=['wav'],
            accept_multiple_files=True,
            key='train_wav'
        )
    
    with col2:
        attack_svl_files = st.file_uploader(
            "Attack Annotation Files (.svl)",
            type=['svl'],
            accept_multiple_files=True,
            key='train_attacks'
        )
    
    with col3:
        sustain_svl_files = st.file_uploader(
            "Sustain Annotation Files (.svl)",
            type=['svl'],
            accept_multiple_files=True,
            key='train_sustain'
        )
    
    # Option to save uploaded files to recordings folder
    st.markdown("---")
    st.subheader("Save Uploads")
    save_to_recordings = st.checkbox(
        "Save uploaded files to recordings folder for future use",
        value=False,
        help="When checked, uploaded files will be saved to the recordings folder so they can be used as sample data in future sessions."
    )
    
    # Load sample data option
    recordings_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'recordings'))
    
    use_sample = st.checkbox("Use sample data from recordings folder", value=False)
    
    if use_sample and os.path.exists(recordings_dir):
        sample_files = os.listdir(recordings_dir)
        wav_samples = [f for f in sample_files if f.endswith('.wav')]
        attack_samples = [f for f in sample_files if 'attack' in f.lower() and f.endswith('.svl')]
        sustain_samples = [f for f in sample_files if 'sustain' in f.lower() and f.endswith('.svl')]
        
        if wav_samples and attack_samples and sustain_samples:
            st.info(f"Found sample data: {wav_samples[0]}")
    
    # Model Configuration Options
    st.markdown("---")
    st.subheader("Model Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Window size for feature extraction
        window_size = st.slider(
            "Feature Extraction Window Size (seconds)",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Size of the window around each onset for extracting audio features. Larger windows capture more context but may blur onset boundaries."
        )
        
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            options=["Random Forest", "Gradient Boosting", "SVM", "Neural Network"],
            index=0,
            help="Select the machine learning algorithm to use for onset classification."
        )
    
    with config_col2:
        st.markdown("**Model Hyperparameters**")
        
        model_params = {}
        
        if model_type == "Random Forest":
            model_params['n_estimators'] = st.slider(
                "Number of Trees",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of decision trees in the forest. More trees generally improve accuracy but increase training time."
            )
            model_params['max_depth'] = st.slider(
                "Max Tree Depth",
                min_value=2,
                max_value=30,
                value=10,
                step=1,
                help="Maximum depth of each tree. Deeper trees can capture more complex patterns but may overfit."
            )
            model_params['min_samples_split'] = st.slider(
                "Min Samples to Split",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                help="Minimum samples required to split a node. Higher values prevent overfitting."
            )
        
        elif model_type == "Gradient Boosting":
            model_params['n_estimators'] = st.slider(
                "Number of Boosting Stages",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of boosting stages. More stages can improve accuracy but increase training time."
            )
            model_params['max_depth'] = st.slider(
                "Max Tree Depth",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Maximum depth of each tree. Gradient boosting typically uses shallow trees."
            )
            model_params['learning_rate'] = st.slider(
                "Learning Rate",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Step size for each boosting iteration. Lower values require more trees but can improve accuracy."
            )
        
        elif model_type == "SVM":
            model_params['C'] = st.slider(
                "Regularization Parameter (C)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Regularization parameter. Higher values create a tighter fit to training data."
            )
            model_params['kernel'] = st.selectbox(
                "Kernel Type",
                options=["rbf", "linear", "poly", "sigmoid"],
                index=0,
                help="Kernel function for the SVM. RBF is a good default for most problems."
            )
            model_params['gamma'] = st.selectbox(
                "Gamma",
                options=["scale", "auto"],
                index=0,
                help="Kernel coefficient. 'scale' uses 1/(n_features * X.var())."
            )
        
        elif model_type == "Neural Network":
            layer1_size = st.slider(
                "First Hidden Layer Size",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                help="Number of neurons in the first hidden layer."
            )
            layer2_size = st.slider(
                "Second Hidden Layer Size",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Number of neurons in the second hidden layer."
            )
            model_params['hidden_layer_sizes'] = (layer1_size, layer2_size)
            model_params['activation'] = st.selectbox(
                "Activation Function",
                options=["relu", "tanh", "logistic"],
                index=0,
                help="Activation function for the hidden layers."
            )
            model_params['learning_rate'] = st.slider(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Initial learning rate for weight updates."
            )
            model_params['max_iter'] = st.slider(
                "Max Iterations",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                help="Maximum number of training iterations."
            )
    
    st.markdown("---")
    
    if st.button("Train Model", type="primary"):
        all_features = []
        uploaded_files_saved = False
        
        # Determine data source
        if use_sample and os.path.exists(recordings_dir):
            # Use sample data
            sample_files = os.listdir(recordings_dir)
            wav_samples = [f for f in sample_files if f.endswith('.wav')]
            attack_samples = [f for f in sample_files if 'attack' in f.lower() and f.endswith('.svl')]
            sustain_samples = [f for f in sample_files if 'sustain' in f.lower() and f.endswith('.svl')]
            
            if wav_samples and attack_samples and sustain_samples:
                with st.spinner("Processing sample data..."):
                    # Load WAV
                    wav_path = os.path.join(recordings_dir, wav_samples[0])
                    y, sr = librosa.load(wav_path, sr=None)
                    
                    # Load attack annotations
                    attack_path = os.path.join(recordings_dir, attack_samples[0])
                    with open(attack_path, 'r') as f:
                        attack_data = parse_svl_file(f.read())
                    attack_times = frames_to_times(attack_data['frames'], attack_data['sample_rate'])
                    
                    # Load sustain annotations
                    sustain_path = os.path.join(recordings_dir, sustain_samples[0])
                    with open(sustain_path, 'r') as f:
                        sustain_data = parse_svl_file(f.read())
                    sustain_times = frames_to_times(sustain_data['frames'], sustain_data['sample_rate'])
                    
                    # Combine onset times and labels
                    onset_times = attack_times + sustain_times
                    labels = [0] * len(attack_times) + [1] * len(sustain_times)  # 0=attack, 1=sustain
                    
                    # Extract features
                    features_df = extract_all_features(y, sr, onset_times, labels, window_size=window_size)
                    if not features_df.empty:
                        all_features.append(features_df)
                    
                    # Show visualization
                    st.subheader("Training Data Visualization")
                    fig = plot_waveform_with_onsets(y, sr, attack_times, sustain_times, 
                                                     title=f"Sample: {wav_samples[0]}")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("Sample data files not found in recordings folder")
        
        elif wav_files and attack_svl_files and sustain_svl_files:
            # Use uploaded files
            # Match files by name (assuming naming convention: filename.wav, filename_attacks.svl, filename_sustain.svl)
            for wav_file in wav_files:
                base_name = wav_file.name.replace('.wav', '')
                
                # Find matching SVL files
                attack_svl = None
                sustain_svl = None
                
                for svl in attack_svl_files:
                    if base_name in svl.name:
                        attack_svl = svl
                        break
                
                for svl in sustain_svl_files:
                    if base_name in svl.name:
                        sustain_svl = svl
                        break
                
                if attack_svl and sustain_svl:
                    with st.spinner(f"Processing {wav_file.name}..."):
                        # Save WAV temporarily and load
                        wav_file.seek(0)
                        wav_content = wav_file.read()
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp.write(wav_content)
                            tmp_path = tmp.name
                        
                        y, sr = librosa.load(tmp_path, sr=None)
                        os.unlink(tmp_path)
                        
                        # Parse SVL files
                        attack_svl.seek(0)
                        attack_content = attack_svl.read()
                        attack_data = parse_svl_file(attack_content)
                        attack_times = frames_to_times(attack_data['frames'], attack_data['sample_rate'])
                        
                        sustain_svl.seek(0)
                        sustain_content = sustain_svl.read()
                        sustain_data = parse_svl_file(sustain_content)
                        sustain_times = frames_to_times(sustain_data['frames'], sustain_data['sample_rate'])
                        
                        # Save to recordings folder if option is selected
                        if save_to_recordings and not uploaded_files_saved:
                            try:
                                if not os.path.exists(recordings_dir):
                                    os.makedirs(recordings_dir)
                                
                                # Save WAV file
                                wav_save_path = os.path.join(recordings_dir, wav_file.name)
                                save_content_to_file(wav_save_path, wav_content)
                                
                                # Save attack SVL file
                                attack_save_path = os.path.join(recordings_dir, attack_svl.name)
                                save_content_to_file(attack_save_path, attack_content)
                                
                                # Save sustain SVL file
                                sustain_save_path = os.path.join(recordings_dir, sustain_svl.name)
                                save_content_to_file(sustain_save_path, sustain_content)
                                
                                st.success(f"✓ Saved {wav_file.name} and annotation files to recordings folder")
                                uploaded_files_saved = True
                            except Exception as e:
                                st.warning(f"Could not save files to recordings folder: {str(e)}")
                        
                        # Combine and extract features
                        onset_times = attack_times + sustain_times
                        labels = [0] * len(attack_times) + [1] * len(sustain_times)
                        
                        features_df = extract_all_features(y, sr, onset_times, labels, window_size=window_size)
                        if not features_df.empty:
                            all_features.append(features_df)
                        
                        # Visualize
                        fig = plot_waveform_with_onsets(y, sr, attack_times, sustain_times,
                                                         title=f"{wav_file.name}")
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.warning("Please upload WAV files and corresponding SVL annotation files, or use sample data.")
        
        # Train if we have features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            
            st.subheader("Training Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(combined_features))
            with col2:
                st.metric("Attack Samples", len(combined_features[combined_features['label'] == 0]))
            with col3:
                st.metric("Sustain Samples", len(combined_features[combined_features['label'] == 1]))
            
            if len(combined_features) < 10:
                st.warning("Not enough samples for reliable training. Consider adding more annotated data.")
            else:
                with st.spinner(f"Training {model_type} model..."):
                    model, scaler, feature_cols, metrics = train_model(
                        combined_features, 
                        model_type=model_type, 
                        model_params=model_params
                    )
                    
                    # Store in session state
                    st.session_state.trained_model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    st.session_state.training_metrics = metrics
                    st.session_state.window_size = window_size
                
                st.success(f"{model_type} model trained successfully!")
                
                # Show metrics
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
                    
                    # Classification report
                    report = metrics['classification_report']
                    report_df = pd.DataFrame({
                        'Class': ['Attack', 'Sustain'],
                        'Precision': [report['Attack']['precision'], report['Sustain']['precision']],
                        'Recall': [report['Attack']['recall'], report['Sustain']['recall']],
                        'F1-Score': [report['Attack']['f1-score'], report['Sustain']['f1-score']]
                    })
                    st.dataframe(report_df)
                
                with col2:
                    # Confusion matrix
                    cm = np.array(metrics['confusion_matrix'])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Attack', 'Sustain'],
                                yticklabels=['Attack', 'Sustain'], ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Feature importance (only for models that support it)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(model, feature_cols)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"Feature importance visualization is not available for {model_type} models.")
                
                # Download model
                st.subheader("Download Trained Model")
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                    joblib.dump({
                        'model': model,
                        'scaler': scaler,
                        'feature_cols': feature_cols,
                        'window_size': window_size,
                        'model_type': model_type
                    }, tmp.name)
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Model (.joblib)",
                            data=f.read(),
                            file_name="onset_detection_model.joblib",
                            mime="application/octet-stream"
                        )
                    os.unlink(tmp.name)


with tab_predict:
    st.header("Predict Onsets in New Recordings")
    
    # Check if model is available
    model_source = st.radio(
        "Model Source",
        ["Use trained model from this session", "Upload saved model"],
        key="model_source"
    )
    
    model_ready = False
    
    if model_source == "Use trained model from this session":
        if st.session_state.trained_model is not None:
            st.success("✓ Model is ready")
            model_ready = True
        else:
            st.warning("No model trained yet. Please train a model first or upload a saved model.")
    else:
        uploaded_model = st.file_uploader("Upload saved model (.joblib)", type=['joblib'])
        if uploaded_model:
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                tmp.write(uploaded_model.read())
                tmp_path = tmp.name
            
            loaded = joblib.load(tmp_path)
            os.unlink(tmp_path)
            
            st.session_state.trained_model = loaded['model']
            st.session_state.scaler = loaded['scaler']
            st.session_state.feature_cols = loaded['feature_cols']
            # Load window_size if available (backward compatibility)
            if 'window_size' in loaded:
                st.session_state.window_size = loaded['window_size']
            model_type_info = loaded.get('model_type', 'Unknown')
            st.success(f"✓ Model loaded successfully ({model_type_info})")
            model_ready = True
    
    if model_ready:
        st.markdown("---")
        st.subheader("Upload Audio for Prediction")
        
        predict_wav = st.file_uploader(
            "WAV Audio File",
            type=['wav'],
            key='predict_wav'
        )
        
        if predict_wav:
            with st.spinner("Processing audio..."):
                # Save and load audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(predict_wav.read())
                    tmp_path = tmp.name
                
                y, sr = librosa.load(tmp_path, sr=None)
                os.unlink(tmp_path)
                
                # Predict using the window_size from session state
                results = predict_onsets(
                    y, sr,
                    st.session_state.trained_model,
                    st.session_state.scaler,
                    st.session_state.feature_cols,
                    window_size=st.session_state.window_size
                )
                
                if results.empty:
                    st.warning("No onsets detected in the audio.")
                else:
                    st.success(f"Detected {len(results)} onsets")
                    
                    # Separate by type
                    attack_times = results[results['predicted_class'] == 'Attack']['onset_time'].tolist()
                    sustain_times = results[results['predicted_class'] == 'Sustain']['onset_time'].tolist()
                    
                    # Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Attack Onsets", len(attack_times))
                    with col2:
                        st.metric("Sustain Onsets", len(sustain_times))
                    
                    # Visualization
                    st.subheader("Detected Onsets Visualization")
                    fig = plot_waveform_with_onsets(y, sr, attack_times, sustain_times,
                                                     title=f"Predictions: {predict_wav.name}")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name=f"onset_predictions_{predict_wav.name.replace('.wav', '')}.csv",
                        mime="text/csv"
                    )


with tab_analyze:
    st.header("Articulation Timing Analysis")
    
    st.markdown("""
    This tool analyzes the **attack duration** — the time between when an articulation starts 
    (attack onset) and when it settles into sustained sound (sustain onset).
    
    Consistent attack durations indicate uniform articulation technique across a performance.
    """)
    
    st.subheader("Upload Annotation Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_attack_svl = st.file_uploader(
            "Attack Annotations (.svl)",
            type=['svl'],
            key='analyze_attacks'
        )
    
    with col2:
        analyze_sustain_svl = st.file_uploader(
            "Sustain Annotations (.svl)",
            type=['svl'],
            key='analyze_sustain'
        )
    
    # Option to use sample data
    recordings_dir_analyze = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'recordings'))
    use_sample_analyze = st.checkbox("Use sample data from recordings folder", value=False, key='use_sample_analyze')
    
    if st.button("Analyze Timing Consistency", type="primary", key='analyze_btn'):
        attack_times = None
        sustain_times = None
        
        if use_sample_analyze and os.path.exists(recordings_dir_analyze):
            sample_files = os.listdir(recordings_dir_analyze)
            attack_samples = [f for f in sample_files if 'attack' in f.lower() and f.endswith('.svl')]
            sustain_samples = [f for f in sample_files if 'sustain' in f.lower() and f.endswith('.svl')]
            
            if attack_samples and sustain_samples:
                with st.spinner("Loading sample annotations..."):
                    attack_path = os.path.join(recordings_dir_analyze, attack_samples[0])
                    with open(attack_path, 'r') as f:
                        attack_data = parse_svl_file(f.read())
                    attack_times = frames_to_times(attack_data['frames'], attack_data['sample_rate'])
                    
                    sustain_path = os.path.join(recordings_dir_analyze, sustain_samples[0])
                    with open(sustain_path, 'r') as f:
                        sustain_data = parse_svl_file(f.read())
                    sustain_times = frames_to_times(sustain_data['frames'], sustain_data['sample_rate'])
                    
                    st.info(f"Using sample data: {attack_samples[0]} and {sustain_samples[0]}")
            else:
                st.error("Sample annotation files not found in recordings folder")
        
        elif analyze_attack_svl and analyze_sustain_svl:
            with st.spinner("Parsing annotation files..."):
                analyze_attack_svl.seek(0)
                attack_data = parse_svl_file(analyze_attack_svl.read())
                attack_times = frames_to_times(attack_data['frames'], attack_data['sample_rate'])
                
                analyze_sustain_svl.seek(0)
                sustain_data = parse_svl_file(analyze_sustain_svl.read())
                sustain_times = frames_to_times(sustain_data['frames'], sustain_data['sample_rate'])
        else:
            st.warning("Please upload both attack and sustain SVL files, or use sample data.")
        
        if attack_times and sustain_times:
            # Pair onsets and calculate durations
            pairs = pair_attack_sustain_onsets(attack_times, sustain_times)
            
            if not pairs:
                st.error("Could not pair attack and sustain onsets. Ensure annotations are properly aligned.")
            else:
                pairs_df = pd.DataFrame(pairs)
                stats, pairs_df = analyze_timing_consistency(pairs_df)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Notes Analyzed", stats['count'])
                with col2:
                    st.metric("Mean Duration", f"{stats['mean_ms']:.1f} ms")
                with col3:
                    st.metric("Std Deviation", f"{stats['std_ms']:.1f} ms")
                with col4:
                    st.metric("Coefficient of Variation", f"{stats['cv_percent']:.1f}%")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Minimum", f"{stats['min_ms']:.1f} ms")
                with col2:
                    st.metric("Maximum", f"{stats['max_ms']:.1f} ms")
                with col3:
                    st.metric("Range", f"{stats['range_ms']:.1f} ms")
                with col4:
                    st.metric("Outliers (>2σ)", stats['outlier_count'])
                
                # Consistency interpretation
                st.subheader("Consistency Assessment")
                cv = stats['cv_percent']
                if cv < 10:
                    st.success(f"✓ **Excellent consistency**: CV = {cv:.1f}% — Attack durations are highly uniform.")
                elif cv < 20:
                    st.info(f"◐ **Good consistency**: CV = {cv:.1f}% — Attack durations are reasonably consistent with some variation.")
                elif cv < 30:
                    st.warning(f"◑ **Moderate consistency**: CV = {cv:.1f}% — Noticeable variation in attack durations.")
                else:
                    st.error(f"✗ **Low consistency**: CV = {cv:.1f}% — Significant variation in attack durations.")
                
                # Visualizations
                st.subheader("Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.markdown("**Distribution of Attack Durations**")
                    fig_hist = plot_timing_histogram(pairs_df, stats)
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
                
                with viz_col2:
                    st.markdown("**Attack Duration Box Plot**")
                    fig_box = plot_timing_boxplot(pairs_df)
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                
                st.markdown("**Attack Duration Over Time**")
                fig_time = plot_timing_over_time(pairs_df, stats)
                st.pyplot(fig_time)
                plt.close(fig_time)
                
                # Detailed data table
                st.subheader("Detailed Note-by-Note Data")
                display_df = pairs_df[['note_index', 'attack_time', 'sustain_time', 'attack_duration_ms', 'is_outlier']].copy()
                display_df.columns = ['Note #', 'Attack Time (s)', 'Sustain Time (s)', 'Attack Duration (ms)', 'Outlier']
                display_df['Attack Time (s)'] = display_df['Attack Time (s)'].round(3)
                display_df['Sustain Time (s)'] = display_df['Sustain Time (s)'].round(3)
                display_df['Attack Duration (ms)'] = display_df['Attack Duration (ms)'].round(2)
                st.dataframe(display_df)
                
                # Download results
                csv = pairs_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Results (CSV)",
                    data=csv,
                    file_name="articulation_timing_analysis.csv",
                    mime="text/csv"
                )


with tab_about:
    st.header("About This Application")
    
    st.markdown("""
    ### Purpose
    This application is designed to help music researchers and clarinetists analyze 
    performance articulations using machine learning. By training on annotated 
    recordings, the model learns to distinguish between different phases of note articulation.
    
    ### Onset Types
    
    **Articulation (Attack) Onsets**
    - The point when an articulation begins
    - Marks the initial transient of the note
    - Characterized by rapid spectral and amplitude changes
    
    **Sustain Onsets**
    - The point when the articulation has settled into sustained sound
    - Marks the transition from attack phase to steady-state
    - Follows the attack onset for the same note
    
    ### Technical Details
    
    **Feature Extraction**
    - Time-domain: RMS energy, zero-crossing rate, attack slope
    - Spectral: Centroid, bandwidth, rolloff, flatness, flux
    - MFCCs: 13 mel-frequency cepstral coefficients (mean and std)
    - Configurable window size for feature extraction (default: 50ms)
    
    **Available Model Types**
    - **Random Forest**: Ensemble of decision trees, good default choice
    - **Gradient Boosting**: Sequential ensemble method, often more accurate
    - **SVM (Support Vector Machine)**: Effective for high-dimensional data
    - **Neural Network (MLP)**: Multi-layer perceptron for complex patterns
    
    **Model Configuration Options**
    - Window size for feature extraction
    - Number of trees/estimators
    - Tree depth and complexity parameters
    - Learning rate (for boosting and neural networks)
    - Kernel type (for SVM)
    - Hidden layer sizes (for neural networks)
    
    ### SVL File Format
    
    The application expects Sonic Visualiser Layer (SVL) files in the following format:
    """)
    
    st.code("""
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE sonic-visualiser>
<sv>
  <data>
    <model id="3" name="" sampleRate="44100" ... />
    <dataset id="2" dimensions="1">
      <point frame="86960" label="New Point" />
      <point frame="110628" label="New Point" />
      ...
    </dataset>
  </data>
  <display>
    <layer id="1" type="timeinstants" ... />
  </display>
</sv>
    """, language="xml")
    
    st.markdown("""
    ### Workflow
    
    1. **Annotate** recordings in Sonic Visualiser, creating separate layers for 
       attack and sustain onsets
    2. **Export** each layer as an SVL file
    3. **Train** the model using the WAV + SVL file pairs (Train Model tab)
    4. **Predict** on new recordings to automatically detect and classify onsets (Predict tab)
    5. **Analyze** timing consistency to evaluate articulation uniformity (Analyze Timing tab)
    
    ### Timing Analysis
    
    The **Analyze Timing** tab calculates the time between each attack onset and its 
    corresponding sustain onset. This "attack duration" measures how long it takes 
    for each note to transition from initial articulation to sustained sound.
    
    **Key Metrics:**
    - **Mean Duration**: Average attack duration across all notes
    - **Standard Deviation**: Measure of variation in attack durations
    - **Coefficient of Variation (CV)**: Relative variability (lower = more consistent)
    - **Outliers**: Notes with unusually long or short attack durations (>2σ from mean)
    
    A low CV indicates consistent articulation technique throughout the performance.
    
    ### Tips for Best Results
    
    - Use consistent annotation criteria across all training files
    - Include diverse examples (different pieces, dynamics, tempos)
    - Aim for at least 50+ examples of each onset type
    - Validate predictions and refine training data as needed
    """)

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
