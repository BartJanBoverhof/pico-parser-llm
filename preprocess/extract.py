from Bio import Entrez, Medline  # Importing necessary modules for PubMed access and parsing.
import re

"""
This script interacts with the PubMed database to retrieve and process scientific articles based on a specified search query. 
It uses the Biopython library to perform the search, fetch detailed article information in MEDLINE format, 
and extract key details such as the title, abstract, DOI, and authors. The results are stored in a dictionary for further analysis.
"""

def fetch_pubmed_articles(query, email, retmax=10):
    Entrez.email = email
    
    # 1. Esearch
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    search_record = Entrez.read(handle)
    handle.close()
    
    pubmed_ids = search_record["IdList"]
    if not pubmed_ids:
        return {}
    
    # 2. Efetch in MEDLINE format
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    
    # 3. Build articles_dict
    articles_dict = {}
    for record in records:
        pmid = record.get("PMID")
        if not pmid:
            continue
        
        # Extract publication year from DP
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
            "PublicationYear": pub_year  # Add the year as metadata
        }

    return articles_dict