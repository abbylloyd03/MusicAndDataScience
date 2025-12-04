import streamlit as st

st.set_page_config(page_title="Data & AI Music Guide", page_icon="🎵", layout="wide")

st.title("The Foundational Guide to Data Science & AI Applications in Music")
st.markdown("""
This app provides introductory tutorials on open-source analytics and AI tools for music generation and analysis. These tutorials are intended to help music students gain fundamental skills needed to understand current AI applications in music.
Explore how to use technologies like music21 for music generation and analysis.

Use the sidebar to navigate to tutorials. Each tutorial includes code examples and links to run them in Google Colab for easy experimentation.
""")

# Sidebar navigation (auto-populated from pages/ folder)
# st.sidebar.title("Navigation")
# st.sidebar.markdown("Select a tutorial below:")

# Footer
st.markdown("---")
st.markdown("©2025, Author: Abby Lloyd.")