##### 0. PRELIMINARIES
import sys
import os
from openai import OpenAI
import json
from PyPDF2 import PdfReader

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd() # Get the current working directory (project root)
if project_dir not in sys.path: 
    sys.path.append(project_dir)

# Import functions
from preprocess.extract import fetch_pubmed_articles
from preprocess.process_pdf import extract_text_from_pdf
from preprocess.process_pdf import process_pdfs

from llm.open_ai import count_tokens
from llm.open_ai import validate_api_key
from llm.open_ai import create_batched_prompts
from llm.open_ai import query_gpt_api_batched


##### 1. DEFINE PARAMETERS
# For pubmed extraction
retmax = 30  # Maximum number of articles to retrieve.
email = "bjboverhof@gmail.com" # Email to use for PubMed API requests.
message = validate_api_key() # Read in, and validate the OpenAI API key.
print(message) # Print the validation message.

# For openai API
openai = OpenAI()
model = "gpt-4o-mini"  # Replace with your model


##### 2. DATA PREPERATION
input_directory = "data/PDF"  # Replace with your actual input directory path
output_directory = "data/text"  # Replace with your desired output directory

process_pdfs(input_directory, output_directory)




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
