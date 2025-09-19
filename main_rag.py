# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS, CONSOLIDATION_CONFIGS
from python.results import run_complete_analysis, ComprehensiveOverview, PICOAnalyzer, OutcomeAnalyzer
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

# Run consolidation for NSCLC test set
print("\n--- Consolidating NSCLC PICOs and Outcomes for Test Set ---")
consolidation_results_nsclc_test = rag_nsclc.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"],
    test_set=True
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

# Run consolidation for HCC (all data goes to test set for HCC)
print("\n--- Consolidating HCC PICOs and Outcomes for Test Set ---")
consolidation_results_hcc_test = rag_hcc.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"],
    test_set=True
)

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================
print("\n" + "="*100)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("="*100)

# First, create comprehensive overview for all cases
print("\n" + "📋 GENERATING COMPREHENSIVE OVERVIEW FOR ALL CASES")
print("="*80)

# Initialize comprehensive overview class
comprehensive_overview = ComprehensiveOverview()

all_pico_files_train = []
all_outcome_files_train = []
all_pico_files_test = []
all_outcome_files_test = []

# Collect all PICO and Outcome files for both train and test sets
for case in ["NSCLC", "HCC"]:
    case_dir = Path(f"results/{case}/consolidated")
    if case_dir.exists():
        # Training files
        train_pico_files = list(case_dir.glob("*consolidated_picos_train*.json"))
        train_outcome_files = list(case_dir.glob("*consolidated_outcomes_train*.json"))
        
        # Test files
        test_pico_files = list(case_dir.glob("*consolidated_picos_test*.json"))
        test_outcome_files = list(case_dir.glob("*consolidated_outcomes_test*.json"))
        
        if train_pico_files and train_outcome_files:
            all_pico_files_train.extend([(max(train_pico_files, key=os.path.getmtime), case)])
            all_outcome_files_train.extend([(max(train_outcome_files, key=os.path.getmtime), case)])
        
        if test_pico_files and test_outcome_files:
            all_pico_files_test.extend([(max(test_pico_files, key=os.path.getmtime), case)])
            all_outcome_files_test.extend([(max(test_outcome_files, key=os.path.getmtime), case)])

# Generate cross-case overviews for both train and test sets
if all_pico_files_train and all_outcome_files_train:
    print("\n--- Generating Training Set Overview ---")
    comprehensive_overview.generate_cross_case_overview(
        all_pico_files_train, 
        all_outcome_files_train,
        output_suffix="_train"
    )

if all_pico_files_test and all_outcome_files_test:
    print("\n--- Generating Test Set Overview ---")
    comprehensive_overview.generate_cross_case_overview(
        all_pico_files_test, 
        all_outcome_files_test,
        output_suffix="_test"
    )

# NSCLC Analysis for both train and test sets
print("\n=== NSCLC DETAILED ANALYSIS ===")
nsclc_consolidated_dir = Path("results/NSCLC/consolidated")
if nsclc_consolidated_dir.exists():
    # Training set analysis
    print("\n--- NSCLC Training Set Analysis ---")
    train_pico_files = list(nsclc_consolidated_dir.glob("*consolidated_picos_train*.json"))
    train_outcome_files = list(nsclc_consolidated_dir.glob("*consolidated_outcomes_train*.json"))
    
    if train_pico_files and train_outcome_files:
        train_pico_file = max(train_pico_files, key=os.path.getmtime)
        train_outcome_file = max(train_outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing NSCLC Training PICO data from: {train_pico_file}")
        print(f"Analyzing NSCLC Training Outcomes data from: {train_outcome_file}")
        print()
        
        try:
            pico_analyzer_nsclc_train, outcome_analyzer_nsclc_train, visualizer_nsclc_train = run_complete_analysis(
                pico_file_path=str(train_pico_file),
                outcome_file_path=str(train_outcome_file),
                output_suffix="_train"
            )
            print("NSCLC training set analysis completed successfully!")
        except Exception as e:
            print(f"Error in NSCLC training set analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: Could not find NSCLC training set consolidated files.")
    
    # Test set analysis
    print("\n--- NSCLC Test Set Analysis ---")
    test_pico_files = list(nsclc_consolidated_dir.glob("*consolidated_picos_test*.json"))
    test_outcome_files = list(nsclc_consolidated_dir.glob("*consolidated_outcomes_test*.json"))
    
    if test_pico_files and test_outcome_files:
        test_pico_file = max(test_pico_files, key=os.path.getmtime)
        test_outcome_file = max(test_outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing NSCLC Test PICO data from: {test_pico_file}")
        print(f"Analyzing NSCLC Test Outcomes data from: {test_outcome_file}")
        print()
        
        try:
            pico_analyzer_nsclc_test, outcome_analyzer_nsclc_test, visualizer_nsclc_test = run_complete_analysis(
                pico_file_path=str(test_pico_file),
                outcome_file_path=str(test_outcome_file),
                output_suffix="_test"
            )
            print("NSCLC test set analysis completed successfully!")
        except Exception as e:
            print(f"Error in NSCLC test set analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: Could not find NSCLC test set consolidated files.")
else:
    print("Warning: results/NSCLC/consolidated directory not found.")
    print("Make sure the NSCLC consolidation step completed successfully.")

# HCC Analysis (test set only since all HCC data goes to test)
print("\n=== HCC DETAILED ANALYSIS ===")
hcc_consolidated_dir = Path("results/HCC/consolidated")
if hcc_consolidated_dir.exists():
    # Test set analysis
    print("\n--- HCC Test Set Analysis ---")
    test_pico_files = list(hcc_consolidated_dir.glob("*consolidated_picos_test*.json"))
    test_outcome_files = list(hcc_consolidated_dir.glob("*consolidated_outcomes_test*.json"))
    
    if test_pico_files and test_outcome_files:
        test_pico_file = max(test_pico_files, key=os.path.getmtime)
        test_outcome_file = max(test_outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing HCC Test PICO data from: {test_pico_file}")
        print(f"Analyzing HCC Test Outcomes data from: {test_outcome_file}")
        print()
        
        try:
            pico_analyzer_hcc_test, outcome_analyzer_hcc_test, visualizer_hcc_test = run_complete_analysis(
                pico_file_path=str(test_pico_file),
                outcome_file_path=str(test_outcome_file),
                output_suffix="_test"
            )
            print("HCC test set analysis completed successfully!")
        except Exception as e:
            print(f"Error in HCC test set analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: Could not find HCC test set consolidated files.")
        if not test_pico_files:
            print("Missing HCC PICO test files in results/HCC/consolidated/")
        if not test_outcome_files:
            print("Missing HCC Outcomes test files in results/HCC/consolidated/")
else:
    print("Warning: results/HCC/consolidated directory not found.")
    print("Make sure the HCC consolidation step completed successfully.")

# Summary statistics for train/test splits
print("\n" + "="*100)
print("TRAIN/TEST SPLIT SUMMARY")
print("="*100)

def print_split_summary(case_name, case_dir):
    """Print summary statistics for train/test split."""
    consolidated_dir = Path(f"results/{case_name}/consolidated")
    if not consolidated_dir.exists():
        print(f"{case_name}: No consolidated directory found")
        return
    
    train_files = len(list(consolidated_dir.glob("*_train_*.json")))
    test_files = len(list(consolidated_dir.glob("*_test_*.json")))
    
    print(f"{case_name}:")
    print(f"  Training files: {train_files}")
    print(f"  Test files: {test_files}")
    
    # Count countries in each split if files exist
    train_pico_files = list(consolidated_dir.glob("*consolidated_picos_train*.json"))
    test_pico_files = list(consolidated_dir.glob("*consolidated_picos_test*.json"))
    
    if train_pico_files:
        try:
            with open(train_pico_files[0], 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            train_countries = train_data.get("consolidation_metadata", {}).get("source_countries", [])
            print(f"  Training countries: {', '.join(train_countries) if train_countries else 'None'}")
        except:
            print(f"  Training countries: Unable to read")
    
    if test_pico_files:
        try:
            with open(test_pico_files[0], 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            test_countries = test_data.get("consolidation_metadata", {}).get("source_countries", [])
            print(f"  Test countries: {', '.join(test_countries) if test_countries else 'None'}")
        except:
            print(f"  Test countries: Unable to read")
    
    print()

import json
print_split_summary("NSCLC", "results/NSCLC")
print_split_summary("HCC", "results/HCC")