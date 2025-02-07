# Insurance Competitive Intelligence RAG

A Retrieval-Augmented Generation (RAG) system for analyzing insurance company financial statements using LanceDB and OpenAI.

## Overview

This project implements a RAG system that allows users to query information from multiple insurance companies' financial statements (10-K reports). It uses LanceDB for vector storage and retrieval, OpenAI for embeddings and question answering, and Streamlit for the user interface.

## Files

- `zesty-rag.ipynb`: Jupyter notebook for data processing and database creationwith simple word-based chunking
- `zesty-rag-langchain_chunking.ipynb`: Jupyter notebook with lancghain chunking
- `zesty-minimal.ipynb`: Jupyter notebook with minimal necessary code to run streamlit app
- `app.py`: Streamlit web application with UI for querying the data
- `all.pdf`: Allstate's 10-K
- `chubb.pdf`: Chubb's 10-K
- `pgr.pdf`: Progressive's 10-K
- `trv.pdf`: Travelers' 10-K

## Setup

1. Install dependencies:

pip install -r requirements.txt

2. Set up your OpenAI API key:
   - Put it directly in the code.

3. Prepare the data:
   - Place PDF files in the projectroot directory
   - Run the Jupyter notebook to process PDFs and create the database:

jupyter notebook rag.ipynb

4. Run the Streamlit app:

streamlit run app.py

# Usage

1. Access the web interface at `http://localhost:8501`
2. Enter your query in the text input field
3. Click "Submit Query" to get results
4. Use example queries from the sidebar for reference

## Data Sources

Financial statements can be obtained from SEC.gov:
1. Visit sec.gov
2. Find the company's 10-K filing
3. View as HTML
4. Print to PDF
5. Save with the appropriate filename (all.pdf, chubb.pdf, pgr.pdf, or trv.pdf)

## License

MIT
