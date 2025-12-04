import streamlit as st
import os

st.set_page_config(page_title="Data & AI Music Guide", page_icon="🎵", layout="wide")

st.title("The Foundational Guide to Data Science & AI Applications in Music")
st.markdown("""
This app provides introductory tutorials on open-source analytics and AI tools for music generation and analysis. These tutorials are intended to help music students gain fundamental skills needed to understand current AI applications in music.
Explore how to use technologies like music21 for music generation and analysis.

Use the sidebar to navigate to tutorials. Each tutorial includes code examples and links to run them in Google Colab for easy experimentation.
""")

# Featured Apps Section
st.markdown("---")
st.header("🎯 Featured Applications")
st.markdown("Explore our interactive music analysis applications:")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images")

# Create two columns for the apps
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎼 MusicXML Pitch-Class Visualizer")
    
    # Display app image
    musicxml_img_path = os.path.join(images_dir, "app_musicxml_visualizer.png")
    if os.path.exists(musicxml_img_path):
        st.image(musicxml_img_path, use_container_width=True)
    
    st.markdown("""
    Upload MusicXML files to visualize pitch patterns with colorful scatter plots.
    
    **Features:**
    - Compare two MusicXML files side-by-side
    - Color notes by pitch class or scale degree
    - Interactive measure selection
    - Custom color palettes
    """)
    
    st.page_link("pages/1_3. App - MusicXML Visualization.py", label="Open MusicXML Visualizer →", icon="🎼")

with col2:
    st.subheader("🎺 Clarinet Onset Detection ML")
    
    # Display app image
    onset_img_path = os.path.join(images_dir, "app_onset_detection.png")
    if os.path.exists(onset_img_path):
        st.image(onset_img_path, use_container_width=True)
    
    st.markdown("""
    Train machine learning models to detect articulation onsets in clarinet recordings.
    
    **Features:**
    - Upload WAV files with annotations
    - Train classification models
    - Predict onsets in new recordings
    - Analyze articulation timing consistency
    """)
    
    st.page_link("pages/1_9._App - Onset Detection ML.py", label="Open Onset Detection ML →", icon="🎺")

# Sidebar navigation (auto-populated from pages/ folder)
# st.sidebar.title("Navigation")
# st.sidebar.markdown("Select a tutorial below:")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")