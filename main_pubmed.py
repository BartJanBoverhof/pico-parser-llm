##### 0. PRELIMINARIES
# Standard library imports
import sys
import os

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd() # Get the current working directory (project root)
if project_dir not in sys.path: 
    sys.path.append(project_dir)

# LLM related imports
from openai import OpenAI

# Local imports
from preprocess.extract import fetch_pubmed_articles

from llm.open_ai import count_tokens
from llm.open_ai import validate_api_key
from llm.open_ai import create_batched_prompts
from llm.open_ai import query_gpt_api_batched

#from preprocess.vectorization import create_vectorstore
from preprocess.vectorization import visualize_vectorstore

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings



##### 1. DEFINE PARAMETERS
# For pubmed extraction
message = validate_api_key() # Read in, and validate the OpenAI API key.
print(message) # Print the validation message.

# For openai API
email = "bjboverhof@gmail.com"
openai = OpenAI()
model = "gpt-4o-mini"  # Replace with your model
embeddings = OpenAIEmbeddings()
db_name = "vector_db"
pubmed_query = "KRAS G12C mutation AND randomized controlled trial OR RCT"  # Query for PubMed search





##### 2. EXTRACT ARTICLES PUBMED
articles = fetch_pubmed_articles(query=pubmed_query, email=email, retmax=100)
articles_keys = list(articles.keys())  # Extract PMIDs from the results.
len(articles_keys)  # Count the number of retrieved articles.
token_count = count_tokens(str(articles)) # Count the number of tokens in the retrieved articles.




import os
import shutil
import re
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from collections import Counter
from Bio import Entrez, Medline
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

def create_vectorstore(docs, db_name="pubmed_vector_db"):
    """
    Creates (and persists) a local Chroma vector store given a list of Document objects.
    """
    if os.path.exists(db_name):
        shutil.rmtree(db_name)
    os.makedirs(db_name, exist_ok=True)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=db_name
    )
    vectorstore.persist()
    
    print(f"Vectorstore created with {vectorstore._collection.count()} documents.")
    return vectorstore

def fetch_pubmed_articles(query, email, retmax=10):
    Entrez.email = email
    
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    search_record = Entrez.read(handle)
    handle.close()
    
    pubmed_ids = search_record["IdList"]
    if not pubmed_ids:
        return {}
    
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    
    articles_dict = {}
    for record in records:
        pmid = record.get("PMID")
        if not pmid:
            continue
        
        dp = record.get("DP", "")
        pub_year = None
        match = re.match(r"(\d{4})", dp)
        if match:
            pub_year = match.group(1)
        
        articles_dict[pmid] = {
            "PMID": pmid,
            "Title": record.get("TI", ""),
            "Abstract": record.get("AB", ""),
            "DOI": record.get("LID", ""),
            "Authors": record.get("AU", []),
            "PublicationYear": pub_year,
            "Journal": record.get("JT", "Unknown")
        }

    return articles_dict

def process_pubmed_for_vectorstore(query, email, retmax=10, db_name="pubmed_vector_db"):
    articles = fetch_pubmed_articles(query, email, retmax=retmax)
    if not articles:
        print("No articles found for the given query.")
        return None

    docs = []
    for pmid, info in articles.items():
        abstract_text = info.get("Abstract", "")
        authors_list = info.get("Authors", [])
        authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else str(authors_list)
        
        metadata = {
            "PMID": pmid,
            "Title": info.get("Title", ""),
            "Authors": authors_str,
            "PublicationYear": info.get("PublicationYear"),
            "DOI": info.get("DOI", ""),
            "Journal": info.get("Journal", "Unknown")
        }
        doc = Document(page_content=abstract_text, metadata=metadata)
        docs.append(doc)

    vectorstore = create_vectorstore(docs, db_name=db_name)
    return vectorstore

def visualize_vectorstore(vectorstore):
    result = vectorstore.get(include=["embeddings", "documents", "metadatas"])
    
    if len(result["embeddings"]) == 0:
        print("No embeddings found in this vector store.")
        return
    
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    metadata_list = result["metadatas"]
    
    doc_journals = [md.get("Journal", "Unknown") if md else "Unknown" for md in metadata_list]
    unique_journals = sorted(set(doc_journals))
    color_palette = ['blue', 'green', 'red', 'orange', 'yellow', 'purple', 'brown', 'pink', 'gray', 'cyan']
    color_map = {journal: color_palette[i % len(color_palette)] for i, journal in enumerate(unique_journals)}
    colors = [color_map[j] for j in doc_journals]
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=6, color=colors, opacity=0.8),
        text=[f"Journal: {j}<br>Text: {doc[:100]}..." if doc else f"Journal: {j}<br>Text: No content"
              for j, doc in zip(doc_journals, documents)],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title='2D Chroma Vector Store Visualization by Journal',
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    fig.show()

vectorstore = process_pubmed_for_vectorstore(
    query="cancer therapy",
    email="your_email@example.com",
    retmax=4000,
    db_name="pubmed_vector_db"
)

visualize_vectorstore(vectorstore)


# If no articles were found or no vector store was created, we can exit
if vectorstore is None:
    print("No articles found or vector store not created.")
else:
    # Otherwise, try a similarity search to see how it performs
    query_string = "Recent therapies for KRAS G12C in lung cancer"
    results = vectorstore.similarity_search(query_string, k=2)  # Get top 2 docs

    print("\n--- Similarity Search Results ---")
    for i, doc in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"Title: {doc.metadata.get('Title')}")
        print(f"PMID: {doc.metadata.get('PMID')}")
        print(f"PublicationYear: {doc.metadata.get('PublicationYear')}")
        print("Abstract excerpt:", doc.page_content[:300], "...")
        print("-" * 50)



### 5. QUERY LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



# --- Function to run the extraction using RAG ---
def run_extraction_per_country(query, vector_store, prompt_template, countries, k):
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Dictionary to store each country’s result
    results_per_country = {}
    
    for country in countries:
        # Each country gets its own retriever
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"country": country}
            }
        )
        
        # Retrieve the top K docs for just this country
        retrieved_for_country = retriever.get_relevant_documents(query)
        
        print(f"--- Retrieved {len(retrieved_for_country)} docs for {country} ---")
        for i, doc in enumerate(retrieved_for_country):
            print(f"Doc {i+1} (country={country}): {doc.page_content}...")
        
        # Build a StuffDocumentsChain and run it for the current country
        stuff_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt_template),
            document_variable_name="context"
        )
        
        country_output = stuff_chain.run(
            input_documents=retrieved_for_country,
            question=query
        )
        
        # Store this country’s result
        results_per_country[country] = country_output

    return results_per_country

# --- Updated System Prompt ---
system_prompt_1 = (
    """
    You are a highly experienced clinical information extraction expert. You have been provided with chunks of clinical guidelines on non-small cell lung cancer with KRAS G12C mutuation from various European countries in different languages.Your task is to extract all medicines, treatments, and therapies that can be considered relevant comparators for an HTA study, along with the corresponding populations for which these treatments may be used, for each country.

    Your answer must be a single table in valid Markdown with three columns: “Country”, “Comparator”, and “Population details.” For country, write the country that is defined in the metadata of the document. For each comparator, create a separate row. If no population details are specified, write ‘No specific details provided.’ Use only the information in the provided context. Do not add extra commentary, explanations, or any text outside of the table. Be as complete as possible.
    """
)

# --- Wrap the system prompt in a LangChain PromptTemplate ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt_1 + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

# Define your query
query = (
    """
    Extract all relevant comparators for an HTA study on non-small cell lung cancer with KRAS G12C mutuation for each European country, along with exact patient population details. Present your findings as instructed, in a single Markdown table with columns for “Country”, “Comparator”, and “Population”
    """
)

# Run the extraction for each country
extraction_results = run_extraction_per_country(
    query=query,
    vector_store=vector_store,
    prompt_template=prompt_template,
    countries=["NL", "EN", "SE", "DE"],
    k=10
)

# Now you have a separate output for each country.
# Decide how you want to handle these separate outputs.
# For example, you can write each one to a separate file:
for country, result in extraction_results.items():
    filename = f"results/output_{country}.md"
    with open(filename, "w") as f:
        f.write(result)
















##### 3. CREATE SYSTEM PROMPT & BATCHED USER PROMPTS
# Define the system prompt
system_prompt = (
    "You are an assistant that analyzes the contents of PubMed abstracts and provides the PICO elements of each specific study. "
    "For each abstract, identify and extract the following components:\n"
    "- **P (Population/Patients)**: Describe the participants and their characteristics.\n"
    "- **I (Intervention)**: The main intervention, treatment, or exposure.\n"
    "- **C (Comparator)**: The control condition or comparison group; if none, state 'Not mentioned'.\n"
    "- **O (Outcome)**: The measured results, endpoints, or key findings relevant to the study.\n\n"
    "Return the extracted PICO elements in plain JSON format, where each PMID is a key, and its value is an object containing the PICO details. "
    "If a component is not mentioned or unclear, use 'Not mentioned'. Do not include any Markdown or code block formatting in your response. Ensure the JSON is well-structured and adheres to proper syntax."
)

# Create batched user prompts 
batched_prompts = create_batched_prompts(articles, batch_size=5)


##### 4. QUERY THE OPENAI API
results = query_gpt_api_batched(batched_prompts, system_prompt, model)


with open('./results/result.json', 'w') as f:
    json.dump(results, f, indent=4)
