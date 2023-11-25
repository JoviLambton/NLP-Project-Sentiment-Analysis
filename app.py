import streamlit as st
import pandas as pd

def process_text(text):
    return text.upper()

def main():
    st.title("Sentiment Analysis Report")

    # File Upload
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])

    if uploaded_file is not None:
        # Read file
        if uploaded_file.type == 'text/plain':
            # Text file
            text = uploaded_file.read()
            st.text("File content:")
            st.text(process_text(text))

if __name__ == "__main__":
    main()
