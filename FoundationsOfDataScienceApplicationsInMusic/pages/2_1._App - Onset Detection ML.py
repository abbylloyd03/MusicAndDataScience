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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt
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
- **Articulation (Attack) Onsets**: Sharp, percussive beginnings of notes
- **Sustain Onsets**: Softer, more gradual note beginnings
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


def extract_all_features(y, sr, onset_times, labels):
    """
    Extract features for all onsets.
    
    Args:
        y: Audio signal
        sr: Sample rate
        onset_times: List of onset times in seconds
        labels: List of labels (0 for attack, 1 for sustain)
        
    Returns:
        DataFrame with features and labels
    """
    all_features = []
    valid_labels = []
    
    for onset_time, label in zip(onset_times, labels):
        features = extract_features_at_onset(y, sr, onset_time)
        if features is not None:
            features['onset_time'] = onset_time
            features['label'] = label
            all_features.append(features)
            valid_labels.append(label)
    
    if all_features:
        return pd.DataFrame(all_features)
    return pd.DataFrame()


# ── Model Training ─────────────────────────────────────────────────
def train_model(features_df):
    """
    Train a Random Forest classifier on the extracted features.
    
    Args:
        features_df: DataFrame with features and 'label' column
        
    Returns:
        tuple (model, scaler, feature_columns, metrics_dict)
    """
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
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': float(np.mean(y_pred == y_test)),
        'classification_report': classification_report(y_test, y_pred, 
                                                        target_names=['Attack', 'Sustain'],
                                                        output_dict=True,
                                                        zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return model, scaler, feature_cols, metrics


def predict_onsets(y, sr, model, scaler, feature_cols, threshold=0.1):
    """
    Detect and classify onsets in audio.
    
    Args:
        y: Audio signal
        sr: Sample rate
        model: Trained classifier
        scaler: Feature scaler
        feature_cols: List of feature column names
        threshold: Onset detection threshold
        
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
        features = extract_features_at_onset(y, sr, onset_time)
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


# ── Session State Initialization ───────────────────────────────────
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.scaler = None
    st.session_state.feature_cols = None
    st.session_state.training_metrics = None


# ── Main Application Tabs ──────────────────────────────────────────
tab_train, tab_predict, tab_about = st.tabs(["🎓 Train Model", "🎯 Predict", "ℹ️ About"])

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
    
    if st.button("Train Model", type="primary"):
        all_features = []
        
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
                    features_df = extract_all_features(y, sr, onset_times, labels)
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
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp.write(wav_file.read())
                            tmp_path = tmp.name
                        
                        y, sr = librosa.load(tmp_path, sr=None)
                        os.unlink(tmp_path)
                        
                        # Parse SVL files
                        attack_svl.seek(0)
                        attack_data = parse_svl_file(attack_svl.read())
                        attack_times = frames_to_times(attack_data['frames'], attack_data['sample_rate'])
                        
                        sustain_svl.seek(0)
                        sustain_data = parse_svl_file(sustain_svl.read())
                        sustain_times = frames_to_times(sustain_data['frames'], sustain_data['sample_rate'])
                        
                        # Combine and extract features
                        onset_times = attack_times + sustain_times
                        labels = [0] * len(attack_times) + [1] * len(sustain_times)
                        
                        features_df = extract_all_features(y, sr, onset_times, labels)
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
                with st.spinner("Training model..."):
                    model, scaler, feature_cols, metrics = train_model(combined_features)
                    
                    # Store in session state
                    st.session_state.trained_model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    st.session_state.training_metrics = metrics
                
                st.success("Model trained successfully!")
                
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
                
                # Feature importance
                st.subheader("Feature Importance")
                fig = plot_feature_importance(model, feature_cols)
                st.pyplot(fig)
                plt.close(fig)
                
                # Download model
                st.subheader("Download Trained Model")
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                    joblib.dump({
                        'model': model,
                        'scaler': scaler,
                        'feature_cols': feature_cols
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
            st.success("✓ Model loaded successfully")
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
                
                # Predict
                results = predict_onsets(
                    y, sr,
                    st.session_state.trained_model,
                    st.session_state.scaler,
                    st.session_state.feature_cols
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


with tab_about:
    st.header("About This Application")
    
    st.markdown("""
    ### Purpose
    This application is designed to help music researchers and clarinetists analyze 
    performance articulations using machine learning. By training on annotated 
    recordings, the model learns to distinguish between different types of note onsets.
    
    ### Onset Types
    
    **Articulation (Attack) Onsets**
    - Characterized by sharp, percussive transients
    - Higher spectral flux and attack slope
    - Common in tongued passages
    
    **Sustain Onsets**
    - Softer, more gradual beginnings
    - Lower attack energy
    - Common in slurred or legato passages
    
    ### Technical Details
    
    **Feature Extraction**
    - Time-domain: RMS energy, zero-crossing rate, attack slope
    - Spectral: Centroid, bandwidth, rolloff, flatness, flux
    - MFCCs: 13 mel-frequency cepstral coefficients (mean and std)
    
    **Model**
    - Random Forest Classifier with 100 trees
    - Balanced class weights for handling imbalanced data
    - StandardScaler normalization
    
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
    3. **Train** the model using the WAV + SVL file pairs
    4. **Predict** on new recordings to automatically detect and classify onsets
    
    ### Tips for Best Results
    
    - Use consistent annotation criteria across all training files
    - Include diverse examples (different pieces, dynamics, tempos)
    - Aim for at least 50+ examples of each onset type
    - Validate predictions and refine training data as needed
    """)

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")
