import streamlit as st
import os
import lancedb
import pyarrow as pa
import pandas as pd
from openai import OpenAI

# Set your OpenAI API key
OPENAI_API_KEY = "PUT YOUR KEY HERE"
client = OpenAI(api_key=OPENAI_API_KEY)

# Use an absolute path for the LanceDB database folder.
DB_URI = os.path.join(os.getcwd(), "lancedb_insurance_competition")
TABLE_NAME = "documents"

# Check if the database folder exists.
if not os.path.exists(DB_URI):
    st.error(f"Database folder not found at {DB_URI}. Please run your notebook to create the database first.")
    st.stop()

# Connect to the existing LanceDB database.
db = lancedb.connect(DB_URI)

# Retrieve the table using dictionary-style indexing.
try:
    table = db[TABLE_NAME]
except Exception as e:
    st.error("Error accessing table. Make sure your notebook has run and created the table.\n"
             f"Error details: {e}")
    st.stop()

# ------------------------------
# RAG Functions (Updated for Multi-Company Queries)
# ------------------------------

def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    response = client.embeddings.create(model=model, input=text)
    embedding = response.data[0].embedding
    return embedding

def extract_companies_from_query(query):
    """
    Extract a list of canonical company identifiers from the query using simple substring matching.
    
    The mapping is as follows:
      - "ALL": ["ALLSTATE"]  (only match "ALLSTATE", not the generic "ALL")
      - "CHUBB": ["CHUBB"]
      - "PGR": ["PGR", "PROGRESSIVE"]
      - "TRV": ["TRV", "TVR", "TRAVELERS", "TRAVELER"]
    
    If a variant is found in the query (case-insensitive), the corresponding canonical code is added.
    """
    query_upper = query.upper()
    company_variants = {
        "ALL": ["ALLSTATE"],
        "CHUBB": ["CHUBB"],
        "PGR": ["PGR", "PROGRESSIVE"],
        "TRV": ["TRV", "TVR", "TRAVELERS", "TRAVELER"],
    }
    found = set()
    for canonical, variants in company_variants.items():
        for variant in variants:
            if variant in query_upper:
                found.add(canonical)
                break
    return list(found)

def search_lancedb(query_embedding, query, k=10):
    """
    Search the LanceDB table for the top k documents similar to the query embedding.
    If the query mentions multiple companies, perform a separate search for each and combine the results.
    """
    companies = extract_companies_from_query(query)
    st.write("Extracted companies from query:", companies)
    
    if companies and len(companies) > 1:
        results_list = []
        for comp in companies:
            st.write(f"Searching for {comp}...")
            res = table.search(query_embedding)\
                       .where(f"company = '{comp}'", prefilter=True)\
                       .limit(k)\
                       .to_pandas()
            st.write(f"Found {len(res)} chunks for {comp}")
            results_list.append(res)
        if results_list:
            results = pd.concat(results_list, ignore_index=True)
        else:
            results = pd.DataFrame()
    elif companies:
        st.write(f"Searching for {companies[0]}...")
        results = table.search(query_embedding)\
                       .where(f"company = '{companies[0]}'", prefilter=True)\
                       .limit(k)\
                       .to_pandas()
        st.write(f"Found {len(results)} chunks for {companies[0]}")
    else:
        st.write("No company filter applied.")
        results = table.search(query_embedding).limit(k).to_pandas()
    return results

def generate_answer(query, retrieved_chunks, model="o3-mini"):
    """
    Generate an answer using the OpenAI chat completions API with markdown formatting.
    """
    context_text = "\n\n".join(retrieved_chunks)
    prompt = f"""You are an insurance competitive intelligence assistant.

Below is context extracted from financial statements of multiple companies.
Please answer the following question by addressing each company mentioned in the question separately.
If context for a company is missing, state that the relevant information is not available.

Format your response in markdown as follows:
1. Start with a brief one-line summary if comparing multiple companies
2. For each company mentioned in the query:
   ## Company Name
   - Key Finding 1
   - Key Finding 2
   - etc.
3. Use markdown headers (##) for company names
4. Use bullet points (-) for listing items
5. Use **bold** for important numbers or metrics
6. Present numerical data consistently
7. Use proper markdown line breaks between sections

Context:
{context_text}

Question:
{query}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in clear, structured analysis. Use markdown formatting for clear presentation."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

def answer_query(query, top_k=10):
    """
    Process the query: compute its embedding, search for relevant document chunks
    (with filtering for each mentioned company), and generate an answer.
    Returns a tuple (answer, retrieved_chunks).
    """
    query_embedding = get_embedding(query)
    results_df = search_lancedb(query_embedding, query, k=top_k)
    if results_df.empty:
        return "No relevant context found for your query.", []
    retrieved_chunks = results_df["text"].tolist() if "text" in results_df.columns else []
    answer = generate_answer(query, retrieved_chunks)
    return answer, retrieved_chunks

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Insurance Competitive Intelligence RAG")

# Sidebar: List all example queries as individual buttons
st.sidebar.header("Example Queries")
example_queries = [
    "What are the causes driving the largest amount of losses across all carriers?",
    "What are the biggest strategic initiatives for Progressive?",
    "Compare Traveler's strategic initiatives with Chubb's.",
    "What is PGR net income?",
    "Compare Chubb, PGR, TRV and Allstate's net revenue for 2023",
]

for q in example_queries:
    if st.sidebar.button(q):
        st.session_state.query_text = q

# Main query input (pre-populated if an example was selected)
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
user_query = st.text_input(
    "Enter your query:",
    value=st.session_state.query_text,
    placeholder="e.g., What are the causes driving the largest amount of losses across all carriers?"
)

# Submit button and response handling
if st.button("Submit Query"):
    with st.spinner("Processing query..."):
        try:
            answer, retrieved_chunks = answer_query(user_query, top_k=10)
            st.markdown("### Answer")
            
            # Display the markdown-formatted answer
            st.markdown(answer)
            
            # Show retrieved chunks in a clean format
            if retrieved_chunks:
                with st.expander("Retrieved Context Chunks"):
                    for idx, chunk in enumerate(retrieved_chunks, 1):
                        preview = chunk[:200] + ("..." if len(chunk) > 200 else "")
                        st.markdown(f"#### Chunk {idx}")
                        st.markdown(preview)
                        st.divider()
                        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")