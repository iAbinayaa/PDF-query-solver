import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Initialize the device
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer for question-answering
model_path = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Function to extract text from PDF
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store
def create_faiss_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Function to answer a question
def answer_question(question, vectorstore):
    # Retrieve relevant documents
    relevant_docs = vectorstore.similarity_search(question)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Format question and context for QA pipeline
    qa_input = {"question": question, "context": context}

    # Get answer from the QA pipeline
    response = qa_pipeline(qa_input)
    return response["answer"]

# Streamlit app
def main():
    st.title("PDF Question-Answering App")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    question = st.text_input("Enter your question")

    if uploaded_file and question:
        with st.spinner("Processing PDF and finding the answer..."):
            # Extract text from PDF
            text = get_pdf_text(uploaded_file)

            # Split text into chunks
            text_chunks = get_text_chunks(text)

            # Create FAISS vector store
            vectorstore = create_faiss_vector_store(text_chunks)

            # Answer the question
            answer = answer_question(question, vectorstore)

            # Display the answer
            st.success("Answer: " + answer)

if __name__ == "__main__":
    main()
