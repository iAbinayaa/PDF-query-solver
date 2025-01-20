# PDF Question-Answering App

This application allows users to upload a PDF file, input a question, and get an answer using a Question-Answering (QA) pipeline powered by Hugging Face Transformers and LangChain libraries. The application is built with Streamlit, providing a user-friendly interface.

---

## Features

- **PDF Text Extraction**: Extracts text from uploaded PDF files.
- **Text Chunking**: Splits the text into manageable chunks for efficient processing.
- **Vector Store**: Uses FAISS (Facebook AI Similarity Search) to store and retrieve relevant text chunks based on the user's query.
- **Question Answering**: Uses a pre-trained model (`distilbert-base-uncased-distilled-squad`) from Hugging Face to answer user queries based on the PDF content.
- **Interactive Interface**: Built with Streamlit for a seamless user experience.

---


## Usage

1. Upload a PDF file by clicking the "Upload a PDF file" button.
2. Enter your question in the input box.
3. View the answer extracted from the content of the PDF.

- **Dependencies**: The app uses the following major Python libraries:
  - `streamlit` for the user interface.
  - `transformers` for the question-answering pipeline.
  - `langchain` and `faiss-cpu` for vector storage and similarity search.
  - `PyPDF2` for extracting text from PDF files.

---

## Model Details

The app uses the following pre-trained models:

1. **Question-Answering Model**: `distilbert-base-uncased-distilled-squad` from Hugging Face Transformers.
2. **Embeddings Model**: `sentence-transformers/all-MiniLM-L12-v2` for creating vector embeddings of the text chunks.

---
