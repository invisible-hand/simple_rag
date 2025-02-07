import streamlit as st
import os
import lancedb
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="ENTER_YOUR_API_KEY")

# Connect to LanceDB
db = lancedb.connect("lancedb_insurance_competition")
table = db["documents"]

def get_embedding(text: str):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

def extract_companies(query):
    """Extract company identifiers from query"""
    query_upper = query.upper()
    company_mapping = {
        "ALL": ["ALL", "ALLSTATE"],
        "CHUBB": ["CHUBB"],
        "PGR": ["PGR", "PROGRESSIVE"],
        "TRV": ["TRV", "TRAVELERS", "TRAVELER"]
    }
    return list({canonical 
                for canonical, variants in company_mapping.items() 
                for variant in variants 
                if variant in query_upper})

def search_documents(query_embedding, query, k=10):
    """Search for relevant documents and return results grouped by company"""
    # If the query explicitly mentions searching all companies, use all.
    if any(phrase in query.lower() for phrase in ["all companies", "all carriers", "all insurers"]):
        companies = ["ALL", "CHUBB", "PGR", "TRV"]
        st.write("📊 Query type: All companies")
    else:
        # Otherwise, try to extract companies from the query.
        companies = extract_companies(query)
        if companies:
            st.write("📊 Query type: Specific companies")
        else:
            # If no companies were detected, default to all companies.
            st.write("📊 No specific company found. Searching across all companies.")
            companies = ["ALL", "CHUBB", "PGR", "TRV"]
    
    st.write("🎯 Companies detected:", ", ".join(companies))
    
    # Perform a separate search for each company and collect the results.
    results = []
    for company in companies:
        company_results = table.search(query_embedding)\
                              .where(f"company = '{company}'", prefilter=True)\
                              .limit(k)\
                              .to_pandas()
        st.write(f"📄 Retrieved {len(company_results)} chunks for {company}")
        results.append(company_results)
    
    # Combine the results into one DataFrame.
    combined_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return combined_results

def generate_answer(query, results_df, model="o3-mini"):
    """Generate answer using retrieved context"""
    # Group context by company and show stats
    context_chunks = []
    st.write("\n💡 Context Details:")
    
    for company, group in results_df.groupby("company"):
        # Use all chunks instead of limiting to top 3
        all_chunks = group["text"].tolist()
        context_chunks.append(f"### {company}\n" + "\n\n".join(all_chunks))
        st.write(f"- {company}: Using all {len(all_chunks)} retrieved chunks")
    
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""Analyze the provided financial statements and answer the question comprehensively.

Context:
{context_text}

Question:
{query}

Please format your response using these guidelines:
1. If comparing multiple companies, start with a brief executive summary (2-3 sentences). Use level 2 markdown headers (##) for the executive summary title.
2. For each company mentioned in the query:
   - Use a level 2 markdown header (##) for the company name
   - Present key findings as bullet points
   - Format numbers as plain text with commas (e.g., "5,246 million")
   - Bold important numbers using markdown (e.g., **5,246 million**)
   - Never use LaTeX notation ($$) or mathematical formatting
   - Spell out percentages in plain text (e.g., "15.7 percent")
3. For financial figures:
   - Always include the unit (million, billion)
   - Use consistent number formatting (e.g., "1,234.5 million")
   - Separate thousands with commas

Example format:
## Company Name
- Net income was **5,246 million** in 2023
- Revenue increased by **15.7 percent** to **10,234 million**"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise financial analyst specializing in the insurance industry. Provide clear analysis using plain text numbers and simple markdown formatting only."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# Custom CSS for better formatting
st.markdown("""
    <style>
    .stMarkdown h2 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        color: #1E88E5;
    }
    .stMarkdown ul {
        padding-bottom: 1rem;
    }
    .stMarkdown li {
        padding-bottom: 0.3rem;
    }
    .stMarkdown strong {
        color: #1E88E5;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Insurance Company Analysis")

# Sidebar with example queries
st.sidebar.header("Example Queries")
example_queries = [
    "What are the causes driving the largest amount of losses across all carriers?",
    "What are the biggest strategic initiatives for Progressive?",
    "Compare Traveler's strategic initiatives with Chubb's.",
    "What is PGR net income?",
    "Compare Chubb, PGR, TRV and Allstate's net income for 2023"
]

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

for ex_query in example_queries:
    if st.sidebar.button(ex_query):
        st.session_state.query_text = ex_query

query = st.text_input("Enter your question:", 
                     value=st.session_state.query_text,
                     placeholder="e.g., What are Progressive's strategic initiatives?")

if st.button("Search"):
    st.markdown("### Query Processing Details")
    
    with st.spinner("Processing..."):
        # Get the query embedding and search for documents.
        query_embedding = get_embedding(query)
        results = search_documents(query_embedding, query)
        
        if results.empty:
            st.warning("No relevant information found.")
        else:
            st.markdown("### Answer")
            answer = generate_answer(query, results)
            st.markdown(answer)
            
            # Display chunk details (metadata only)
            st.markdown("### Retrieved Chunk Metadata by Company")
            for company, group in results.groupby("company"):
                with st.expander(f"{company} - {len(group)} chunks"):
                    for idx, (_, row) in enumerate(group.iterrows(), 1):
                        st.markdown(f"**Chunk {idx} (ID: {row['id']})**")
                        st.markdown(f"*Source:* {row['source']}")
                        if '_distance' in row:
                            st.markdown(f"*Relevancy Score:* {row['_distance']:.4f}")
                        # Show only the first 5 lines of the chunk text
                        chunk_preview = "\n".join(row['text'].splitlines()[:5])
                        st.markdown(f"*Preview:* {chunk_preview}")
                        st.divider()
