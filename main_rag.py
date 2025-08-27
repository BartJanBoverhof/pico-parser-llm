# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS

# Define paths
PDF_PATH = "data/PDF"
CLEAN_PATH = "data/text_cleaned"
TRANSLATED_PATH = "data/text_translated"
POST_CLEANED_PATH = "data/post_cleaned"
CHUNKED_PATH = "data/text_chunked"
VECTORSTORE_PATH = "data/vectorstore"
VECTORSTORE_TYPE = "biobert"  # Choose between "openai", "biobert", or "both"
MODEL = "gpt-4o-mini"
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
    source_type_configs=SOURCE_TYPE_CONFIGS
)

# Load the vectorstore
rag.vectorize_documents(embeddings_type=VECTORSTORE_TYPE)

# Initialize the retriever with the created vectorstore
rag.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE)

# Test Case 1: NSCLC with KRAS G12C mutation
print("\n--- Testing Case 1: NSCLC KRAS G12C Split Retrieval ---")
case1_config = CASE_CONFIGS["case_1_nsclc_krasg12c_monotherapy_progressed"]
case1_indication = case1_config["indication"]

# Get case-specific parameters
case1_required_terms = case1_config.get("required_terms_clinical")
case1_mutation_boost = case1_config.get("mutation_boost_terms", [])

# Step 7: Run SPLIT retrieval for Case 1 (saves chunks to separate files)
print("\n--- Running Case 1 Split Retrieval Step ---")

# Run HTA Population & Comparator retrieval for Case 1
print("Running HTA Population & Comparator retrieval for Case 1...")
rag.run_population_comparator_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=50,
    final_k=20
)

# Run HTA Outcomes retrieval for Case 1
print("Running HTA Outcomes retrieval for Case 1...")
rag.run_outcomes_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=40,
    final_k=15
)

# Run Clinical Guideline Population & Comparator retrieval for Case 1
print("Running Clinical Guideline Population & Comparator retrieval for Case 1...")
rag.run_population_comparator_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=70,
    final_k=18
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


"""
# Alternative: Run both Population & Comparator and Outcomes retrieval in one call
print("\n--- Alternative: Running Case 1 Combined Split Retrieval ---")

# Run complete split retrieval for HTA submissions
print("Running complete HTA split retrieval for Case 1...")
hta_split_results = rag.run_split_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k_pc=50,
    final_k_pc=20,
    initial_k_outcomes=40,
    final_k_outcomes=15
)

# Run complete split retrieval for Clinical Guidelines
print("Running complete Clinical Guideline split retrieval for Case 1...")
clinical_split_results = rag.run_split_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k_pc=70,
    final_k_pc=18,
    initial_k_outcomes=60,
    final_k_outcomes=12
)
"""

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

# Demonstration: Run case-based pipeline with split retrieval
print("\n--- Running Case 1 with Complete Split Retrieval Pipeline ---")
case1_split_pipeline_results = rag.run_case_based_pipeline_with_split_retrieval(
    case_config=case1_config,
    countries=COUNTRIES,
    source_types=["hta_submission", "clinical_guideline"],
    initial_k_pc=50,
    final_k_pc=20,
    initial_k_outcomes=40,
    final_k_outcomes=15,
    skip_processing=True,
    skip_translation=True
)

# For comparison: Show both individual calls and combined split retrieval
print("\n--- Individual Split Retrieval Calls vs Combined ---")

# Method 1: Individual calls (as shown above)
print("✓ Method 1: Individual split retrieval calls completed")

# Method 2: Combined split retrieval call
print("✓ Method 2: Combined split retrieval calls completed")

# Demonstration: Run case-based pipeline with split retrieval
print("\n--- Running Case 1 with Complete Split Retrieval Pipeline ---")
case1_split_pipeline_results = rag.run_case_based_pipeline_with_split_retrieval(
    case_config=case1_config,
    countries=COUNTRIES,
    source_types=["hta_submission", "clinical_guideline"],
    initial_k_pc=50,
    final_k_pc=20,
    initial_k_outcomes=40,
    final_k_outcomes=15,
    skip_processing=True,
    skip_translation=True
)

# Summary
print("\n=== SPLIT RETRIEVAL PIPELINE EXECUTION SUMMARY ===")
print("✓ Documents processed and vectorized")
print("✓ Split retrieval system initialized with specialized capabilities")
print("✓ Population & Comparator chunk retrieval completed and saved")
print("✓ Outcomes chunk retrieval completed and saved")
print("✓ PICO extraction completed using split retrieval chunks")
print(f"✓ Case 1 HTA submissions: {len(extracted_picos_hta_case1)} countries processed")
print(f"✓ Case 1 Clinical guidelines: {len(extracted_picos_clinical_case1)} countries processed")
print(f"✓ Model used: {MODEL}")
print(f"✓ Vectorstore: {VECTORSTORE_TYPE}")
print("✓ Split retrieval approach successfully implemented")

# Print file locations for split retrieval
print("\n=== SPLIT RETRIEVAL OUTPUT FILES ===")
print("📁 HTA Population & Comparator chunks: results/chunks/hta_submission_population_comparator_*_retrieval_results.json")
print("📁 HTA Outcomes chunks: results/chunks/hta_submission_outcomes_*_retrieval_results.json")
print("📁 Clinical Guideline Population & Comparator chunks: results/chunks/clinical_guideline_population_comparator_*_retrieval_results.json")
print("📁 Clinical Guideline Outcomes chunks: results/chunks/clinical_guideline_outcomes_*_retrieval_results.json")
print("📁 HTA submission PICOs: results/PICO/hta_submission_picos.json")
print("📁 Clinical guideline PICOs: results/PICO/clinical_guideline_picos.json")

# Print comparison of split retrieval methods
print("\n=== SPLIT RETRIEVAL METHOD COMPARISON ===")
if hta_split_results and 'population_comparator' in hta_split_results:
    pc_chunks = sum(len(chunks) for chunks in hta_split_results['population_comparator']['chunks_by_country'].values())
    outcomes_chunks = sum(len(chunks) for chunks in hta_split_results['outcomes']['chunks_by_country'].values())
    print(f"📊 Individual Calls - HTA Population & Comparator chunks: {pc_chunks}")
    print(f"📊 Individual Calls - HTA Outcomes chunks: {outcomes_chunks}")
    print(f"📊 Individual Calls - Total HTA chunks: {pc_chunks + outcomes_chunks}")
    print(f"📊 Combined Split Call - Same results as individual calls")

print("\n=== NEXT STEPS ===")
print("1. Review the separate chunk files to ensure proper separation of Population & Comparator vs Outcomes")
print("2. Analyze PICO extraction results to validate that both types of information are captured")
print("3. Compare different split retrieval parameter combinations for optimal results")
print("4. Adjust retrieval parameters (initial_k, final_k) based on chunk quality and coverage")
print("5. Fine-tune query templates in config.py for better separation of concerns")
print("6. Test the pipeline with different case configurations and indications")