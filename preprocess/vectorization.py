import glob
import os
import numpy as np
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from sklearn.manifold import TSNE
import plotly.graph_objects as go


def chunker(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits a list of Document objects into smaller chunks using the
    RecursiveCharacterTextSplitter. This is most useful for longer texts.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", "-", ","]
    )

    # Apply chunking
    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks after chunking: {len(chunks)}")
    return chunks



def create_vectorstore(docs, db_name="pubmed_vector_db"):
    """
    Creates (and persists) a local Chroma vector store given a list of Document objects.
    """
    # Ensure we have a clean directory
    if os.path.exists(db_name):
        shutil.rmtree(db_name)
    os.makedirs(db_name, exist_ok=True)

    embeddings = OpenAIEmbeddings()

    # Create the vectorstore with docs
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=db_name
    )
    vectorstore.persist()
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

    return vectorstore




import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def visualize_vectorstore(vectorstore):
    """
    Visualize the embeddings in a Chroma vector store by reducing them to 2D 
    with t-SNE and color-coding the points by PublicationYear (or another metadata).
    
    Requirements:
      - pip install scikit-learn plotly
      - vectorstore must be a Chroma object with embedded documents
    """
    # 1. Retrieve all embeddings, documents, and metadata
    result = vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    metadata_list = result["metadatas"]
    
    if len(vectors) == 0:
        print("No embeddings found in this vector store.")
        return
    
    # 2. Extract a chosen metadata field for color-coding. 
    #    We'll use 'PublicationYear' if it exists, otherwise 'Unknown'.
    doc_years = [md.get("PublicationYear", "Unknown") for md in metadata_list]
    
    # 3. Assign a color to each unique year
    unique_years = sorted(set(doc_years))
    color_palette = ['blue', 'green', 'red', 'orange', 'yellow', 
                     'purple', 'brown', 'pink', 'gray', 'cyan']
    color_map = {}
    for i, year in enumerate(unique_years):
        color_map[year] = color_palette[i % len(color_palette)]
    
    colors = [color_map[y] for y in doc_years]
    
    # 4. Use t-SNE to reduce vector dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # 5. Build the scatter plot
    #    The hover text shows the publication year and first 100 characters of the document text.
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=6, color=colors, opacity=0.8),
        text=[f"Year: {y}<br>Text: {doc[:100]}..." 
              for y, doc in zip(doc_years, documents)],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title='2D Chroma Vector Store Visualization',
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    fig.show()

