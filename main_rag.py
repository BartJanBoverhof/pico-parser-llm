# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS, CONSOLIDATION_CONFIGS
from python.results import run_complete_analysis
import glob
import os
from pathlib import Path

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

# Define cases to process
CASES = ["nsclc", "hcc"]

# Validate OpenAI API key
validate_api_key()

# Show folder structure
tree = FolderTree(root_path=".")
tree.generate()

# ============================================================================
# COMMON PREPROCESSING STEPS (Done once for all cases)
# ============================================================================

print("\n" + "="*80)
print("COMMON PREPROCESSING STEPS")
print("="*80)

# Step 1: Process PDFs
print("\n=== Step 1: Processing PDFs ===")
PDFProcessor.process_pdfs(
    input_dir=PDF_PATH,
    output_dir=CLEAN_PATH
)

# Step 2: Translate documents
print("\n=== Step 2: Translating Documents ===")
translator = Translator(
    input_dir=CLEAN_PATH,
    output_dir=TRANSLATED_PATH
)
translator.translate_documents()

# Step 3: Clean translated documents 
print("\n=== Step 3: Post-Cleaning Translated Documents ===")
cleaner = PostCleaner(
    input_dir=TRANSLATED_PATH,
    output_dir=POST_CLEANED_PATH,
    maintain_folder_structure=True
)
cleaner.clean_all_documents()

# Step 4: Chunk documents (use cleaned translations)
print("\n=== Step 4: Chunking Documents ===")

chunker = Chunker(
    json_folder_path=POST_CLEANED_PATH,
    output_dir=CHUNKED_PATH,
    chunk_size=600,
    maintain_folder_structure=True
)
chunker.run_pipeline()

print("\n" + "="*80)
print("COMMON PREPROCESSING COMPLETED")
print("="*80)

# ============================================================================
# CASE-SPECIFIC PROCESSING: NSCLC
# ============================================================================
# Step 5: Vectorize NSCLC documents
print("\n=== Step 5: Creating NSCLC Vectorstore ===")
vectoriser_nsclc = Vectoriser(
    chunked_folder_path=CHUNKED_PATH,
    embedding_choice=VECTORSTORE_TYPE,
    db_parent_dir=VECTORSTORE_PATH,
    case="nsclc"
)
vectorstore_nsclc = vectoriser_nsclc.run_pipeline()

# Step 6: Initialize NSCLC RAG system
print("\n=== Step 6: Initializing NSCLC RAG System ===")
rag_nsclc = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE,
    case="nsclc",
    source_type_configs=SOURCE_TYPE_CONFIGS,
    consolidation_configs=CONSOLIDATION_CONFIGS,
    chunked_path=CHUNKED_PATH,
    vectorstore_path=VECTORSTORE_PATH
)

# Load the NSCLC vectorstore
rag_nsclc.vectorize_documents(embeddings_type=VECTORSTORE_TYPE, case="nsclc")

# Initialize the retriever with the NSCLC vectorstore
rag_nsclc.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE, case="nsclc")

# Get NSCLC case configuration
print("\n=== Step 7: Loading NSCLC Case Configuration ===")
case1_config = CASE_CONFIGS["case_1_nsclc_krasg12c_monotherapy_progressed"]
case1_indication = case1_config["indication"]
case1_required_terms = case1_config.get("required_terms_clinical")
case1_mutation_boost = case1_config.get("mutation_boost_terms", [])

# Step 8: Run retrieval for NSCLC Case
print("\n=== Step 8: Running NSCLC Retrieval ===")

# Run HTA Population & Comparator retrieval for NSCLC
print("\n--- Running HTA Population & Comparator retrieval for NSCLC ---")
result_nsclc_hta_pc = rag_nsclc.run_population_comparator_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=15
)

# Run HTA Outcomes retrieval for NSCLC
print("\n--- Running HTA Outcomes retrieval for NSCLC ---")
result_nsclc_hta_outcomes = rag_nsclc.run_outcomes_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case1_indication,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=15
)

# Run Clinical Guideline Population & Comparator retrieval for NSCLC
print("\n--- Running Clinical Guideline Population & Comparator retrieval for NSCLC ---")
result_nsclc_clinical_pc = rag_nsclc.run_population_comparator_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=12
)

# Run Clinical Guideline Outcomes retrieval for NSCLC
print("\n--- Running Clinical Guideline Outcomes retrieval for NSCLC ---")
result_nsclc_clinical_outcomes = rag_nsclc.run_outcomes_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case1_indication,
    required_terms=case1_required_terms,
    mutation_boost_terms=case1_mutation_boost,
    initial_k=60,
    final_k=12
)

# Step 9: Initialize NSCLC PICO extractors
print("\n=== Step 9: Initializing NSCLC PICO Extractors ===")
rag_nsclc.initialize_pico_extractors()

# Step 10: Run PICO extraction for NSCLC
print("\n=== Step 10: Running NSCLC PICO Extraction ===")

# Extract PICOs from HTA submissions for NSCLC
print("\n--- Extracting NSCLC HTA Submission PICOs ---")
extracted_picos_hta_nsclc = rag_nsclc.run_pico_extraction_for_source_type(
    source_type="hta_submission",
    indication=case1_indication
)

# Extract PICOs from clinical guidelines for NSCLC
print("\n--- Extracting NSCLC Clinical Guideline PICOs ---")
extracted_picos_clinical_nsclc = rag_nsclc.run_pico_extraction_for_source_type(
    source_type="clinical_guideline",
    indication=case1_indication
)

# Step 11: Run PICO and Outcomes Consolidation for NSCLC
print("\n=== Step 11: Running NSCLC PICO and Outcomes Consolidation ===")

# Initialize the NSCLC consolidator
rag_nsclc.initialize_pico_consolidator()

# Run consolidation for both source types for NSCLC
print("\n--- Consolidating NSCLC PICOs and Outcomes across all sources ---")
consolidation_results_nsclc = rag_nsclc.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"]
)

# ============================================================================
# CASE-SPECIFIC PROCESSING: HCC
# ============================================================================
# Step 5: Vectorize HCC documents
print("\n=== Step 5: Creating HCC Vectorstore ===")
vectoriser_hcc = Vectoriser(
    chunked_folder_path=CHUNKED_PATH,
    embedding_choice=VECTORSTORE_TYPE,
    db_parent_dir=VECTORSTORE_PATH,
    case="hcc"
)

# Step 6: Initialize HCC RAG system
print("\n=== Step 6: Initializing HCC RAG System ===")
rag_hcc = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE,
    case="hcc",
    source_type_configs=SOURCE_TYPE_CONFIGS,
    consolidation_configs=CONSOLIDATION_CONFIGS,
    chunked_path=CHUNKED_PATH,
    vectorstore_path=VECTORSTORE_PATH
)

# Load the HCC vectorstore
rag_hcc.vectorize_documents(embeddings_type=VECTORSTORE_TYPE, case="hcc")

# Initialize the retriever with the HCC vectorstore
rag_hcc.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE, case="hcc")

# Step 7: Load HCC case configuration
print("\n=== Step 7: Loading HCC Case Configuration ===")
case2_config = CASE_CONFIGS["case_2_hcc_advanced_unresectable"]
case2_indication = case2_config["indication"]
case2_required_terms = case2_config.get("required_terms_clinical")
case2_mutation_boost = case2_config.get("mutation_boost_terms", [])

# Step 8: Run retrieval for HCC Case
print("\n=== Step 8: Running HCC Retrieval ===")

# Run HTA Population & Comparator retrieval for HCC
print("\n--- Running HTA Population & Comparator retrieval for HCC ---")
result_hcc_hta_pc = rag_hcc.run_population_comparator_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case2_indication,
    mutation_boost_terms=case2_mutation_boost,
    initial_k=60,
    final_k=15
)

# Run HTA Outcomes retrieval for HCC
print("\n--- Running HTA Outcomes retrieval for HCC ---")
result_hcc_hta_outcomes = rag_hcc.run_outcomes_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=case2_indication,
    mutation_boost_terms=case2_mutation_boost,
    initial_k=60,
    final_k=15
)

# Run Clinical Guideline Population & Comparator retrieval for HCC
print("\n--- Running Clinical Guideline Population & Comparator retrieval for HCC ---")
result_hcc_clinical_pc = rag_hcc.run_population_comparator_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case2_indication,
    required_terms=case2_required_terms,
    mutation_boost_terms=case2_mutation_boost,
    initial_k=60,
    final_k=12
)

# Run Clinical Guideline Outcomes retrieval for HCC
print("\n--- Running Clinical Guideline Outcomes retrieval for HCC ---")
result_hcc_clinical_outcomes = rag_hcc.run_outcomes_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=case2_indication,
    required_terms=case2_required_terms,
    mutation_boost_terms=case2_mutation_boost,
    initial_k=60,
    final_k=12
)

# Step 9: Initialize HCC PICO extractors
print("\n=== Step 9: Initializing HCC PICO Extractors ===")
rag_hcc.initialize_pico_extractors()

# Step 10: Run PICO extraction for HCC
print("\n=== Step 10: Running HCC PICO Extraction ===")

# Extract PICOs from HTA submissions for HCC
print("\n--- Extracting HCC HTA Submission PICOs ---")
extracted_picos_hta_hcc = rag_hcc.run_pico_extraction_for_source_type(
    source_type="hta_submission",
    indication=case2_indication
)

# Extract PICOs from clinical guidelines for HCC
print("\n--- Extracting HCC Clinical Guideline PICOs ---")
extracted_picos_clinical_hcc = rag_hcc.run_pico_extraction_for_source_type(
    source_type="clinical_guideline",
    indication=case2_indication
)

# Step 11: Run PICO and Outcomes Consolidation for HCC
print("\n=== Step 11: Running HCC PICO and Outcomes Consolidation ===")

# Initialize the HCC consolidator
rag_hcc.initialize_pico_consolidator()

# Run consolidation for both source types for HCC
print("\n--- Consolidating HCC PICOs and Outcomes across all sources ---")
consolidation_results_hcc = rag_hcc.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"]
)

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

# NSCLC Analysis
print("\n=== NSCLC Analysis ===")
nsclc_consolidated_dir = Path("results/NSCLC/consolidated")
if nsclc_consolidated_dir.exists():
    # Get the most recent PICO and Outcomes files for NSCLC
    pico_files = list(nsclc_consolidated_dir.glob("*consolidated_picos*.json"))
    outcome_files = list(nsclc_consolidated_dir.glob("*consolidated_outcomes*.json"))
    
    if pico_files and outcome_files:
        # Sort by modification time and get the most recent
        pico_file = max(pico_files, key=os.path.getmtime)
        outcome_file = max(outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing NSCLC PICO data from: {pico_file}")
        print(f"Analyzing NSCLC Outcomes data from: {outcome_file}")
        print()
        
        try:
            # Run comprehensive analysis for NSCLC
            pico_analyzer_nsclc, outcome_analyzer_nsclc, visualizer_nsclc = run_complete_analysis(
                pico_file_path=str(pico_file),
                outcome_file_path=str(outcome_file)
            )
            print("NSCLC analysis completed successfully!")
        except Exception as e:
            print(f"Error in NSCLC analysis: {e}")
            
    else:
        print("Warning: Could not find consolidated NSCLC PICO or Outcomes files.")
        if not pico_files:
            print("Missing NSCLC PICO files in results/NSCLC/consolidated/")
        if not outcome_files:
            print("Missing NSCLC Outcomes files in results/NSCLC/consolidated/")
            
else:
    print("Warning: results/NSCLC/consolidated directory not found.")
    print("Make sure the NSCLC consolidation step completed successfully.")

# HCC Analysis
print("\n=== HCC Analysis ===")
hcc_consolidated_dir = Path("results/HCC/consolidated")
if hcc_consolidated_dir.exists():
    # Get the most recent PICO and Outcomes files for HCC
    pico_files = list(hcc_consolidated_dir.glob("*consolidated_picos*.json"))
    outcome_files = list(hcc_consolidated_dir.glob("*consolidated_outcomes*.json"))
    
    if pico_files and outcome_files:
        # Sort by modification time and get the most recent
        pico_file = max(pico_files, key=os.path.getmtime)
        outcome_file = max(outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing HCC PICO data from: {pico_file}")
        print(f"Analyzing HCC Outcomes data from: {outcome_file}")
        print()
        
        try:
            # Run comprehensive analysis for HCC
            pico_analyzer_hcc, outcome_analyzer_hcc, visualizer_hcc = run_complete_analysis(
                pico_file_path=str(pico_file),
                outcome_file_path=str(outcome_file)
            )
            print("HCC analysis completed successfully!")
        except Exception as e:
            print(f"Error in HCC analysis: {e}")
            
    else:
        print("Warning: Could not find consolidated HCC PICO or Outcomes files.")
        if not pico_files:
            print("Missing HCC PICO files in results/HCC/consolidated/")
        if not outcome_files:
            print("Missing HCC Outcomes files in results/HCC/consolidated/")
            
else:
    print("Warning: results/HCC/consolidated directory not found.")
    print("Make sure the HCC consolidation step completed successfully.")
