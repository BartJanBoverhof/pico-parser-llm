# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS, CONSOLIDATION_CONFIGS

# Define paths
PDF_PATH = "data/PDF"
CLEAN_PATH = "data/text_cleaned"
TRANSLATED_PATH = "data/text_translated"
POST_CLEANED_PATH = "data/post_cleaned"
CHUNKED_PATH = "data/text_chunked"
VECTORSTORE_PATH = "data/vectorstore"
VECTORSTORE_TYPE = "biobert"  # Choose between "openai", "biobert", or "both"
MODEL = "gpt-4.1"
#MODEL = "gpt-4o-mini"
COUNTRIES = ["ALL"]  # Use "ALL" to process all available countries

# Validate OpenAI API key
validate_api_key()

# Show folder structure
tree = FolderTree(root_path=".")
tree.generate()

# Step 1: Process PDFs
PDFProcessor.process_pdfs(
    input_dir=PDF_PATH,
    output_dir=CLEAN_PATH
)

# Step 2: Translate documents
translator = Translator(
    input_dir=CLEAN_PATH,
    output_dir=TRANSLATED_PATH
)
translator.translate_documents()

# Step 3: Clean translated documents 
cleaner = PostCleaner(
    input_dir=TRANSLATED_PATH,
    output_dir=POST_CLEANED_PATH,
    maintain_folder_structure=True
)
cleaner.clean_all_documents()

# Step 4: Chunk documents (use cleaned translations)
chunker = Chunker(
    json_folder_path=POST_CLEANED_PATH,
    output_dir=CHUNKED_PATH,
    chunk_size=600,
    maintain_folder_structure=True
)
chunker.run_pipeline()

# Step 5: Vectorize documents (creates a unified vectorstore)
vectoriser = Vectoriser(
    chunked_folder_path=CHUNKED_PATH,
    embedding_choice=VECTORSTORE_TYPE,
    db_parent_dir=VECTORSTORE_PATH
)
vectorstore = vectoriser.run_pipeline()

# Step 6: Initialize enhanced RAG system for retrieval and LLM querying
rag = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE,
    source_type_configs=SOURCE_TYPE_CONFIGS,
    consolidation_configs=CONSOLIDATION_CONFIGS
)

# Load the vectorstore
rag.vectorize_documents(embeddings_type=VECTORSTORE_TYPE)

# Initialize the retriever with the created vectorstore
rag.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE)

# Test Case 1: NSCLC with KRAS G12C mutation
print("\n--- Testing Case 1: NSCLC KRAS G12C Retrieval & Extraction ---")
case1_config = CASE_CONFIGS["case_1_nsclc_krasg12c_monotherapy_progressed"]
case1_indication = case1_config["indication"]

# Get case-specific parameters
case1_required_terms = case1_config.get("required_terms_clinical")
case1_mutation_boost = case1_config.get("mutation_boost_terms", [])

# Step 7: Run retrieval for Case 1 (saves chunks to separate files)
print("\n--- Running Case 1 Retrieval Step ---")

# Run HTA Population & Comparator retrieval for Case 1
print("Running HTA Population & Comparator retrieval for Case 1...")
rag.run_population_comparator_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=22
)

# Run HTA Outcomes retrieval for Case 1
print("Running HTA Outcomes retrieval for Case 1...")
rag.run_outcomes_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=22
)

# Run Clinical Guideline Population & Comparator retrieval for Case 1
print("Running Clinical Guideline Population & Comparator retrieval for Case 1...")
rag.run_population_comparator_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=12
)

# Run Clinical Guideline Outcomes retrieval for Case 1
print("Running Clinical Guideline Outcomes retrieval for Case 1...")
rag.run_outcomes_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=12
)

# Step 8: Initialize PICO extractors
rag.initialize_pico_extractors()

# Step 9: Run PICO extraction for Case 1 (uses stored chunks from results/chunks)
print("\n--- Running Case 1 PICO Extraction Step ---")

# Extract PICOs from HTA submissions for Case 1
print("Extracting Case 1 HTA Submission PICOs...")
extracted_picos_hta_case1 = rag.run_pico_extraction_for_source_type(
    source_type="hta_submission",
    indication=case1_indication
)

# Extract PICOs from clinical guidelines for Case 1
print("Extracting Case 1 Clinical Guideline PICOs...")
extracted_picos_clinical_case1 = rag.run_pico_extraction_for_source_type(
    source_type="clinical_guideline",
    indication=case1_indication
)

# Step 10: Run PICO and Outcomes Consolidation
print("\n--- Running Case 1 PICO and Outcomes Consolidation ---")

# Initialize the consolidator
rag.initialize_pico_consolidator()

# Run consolidation for both source types
print("Consolidating PICOs and Outcomes across all sources...")
consolidation_results = rag.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"]
)
