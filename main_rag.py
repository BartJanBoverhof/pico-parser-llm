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
CASES = ["NSCLC", "HCC"]  # Add your case identifiers here

# Validate OpenAI API key
validate_api_key()

# Show folder structure
tree = FolderTree(root_path=".")
tree.generate()

# Step 1: Process PDFs (if needed - usually done once for all cases)
print("\n=== Step 1: PDF Processing ===")
PDFProcessor.process_pdfs(
    input_dir=PDF_PATH,
    output_dir=CLEAN_PATH
)

# Step 2: Translate documents (if needed - usually done once for all cases)
print("\n=== Step 2: Translation ===")
translator = Translator(
    input_dir=CLEAN_PATH,
    output_dir=TRANSLATED_PATH
)
translator.translate_documents()

# Step 3: Clean translated documents (if needed - usually done once for all cases)
print("\n=== Step 3: Post-Cleaning ===")
cleaner = PostCleaner(
    input_dir=TRANSLATED_PATH,
    output_dir=POST_CLEANED_PATH,
    maintain_folder_structure=True
)
cleaner.clean_all_documents()

# Step 4: Chunk documents (if needed - usually done once for all cases)
print("\n=== Step 4: Chunking ===")
chunker = Chunker(
    json_folder_path=POST_CLEANED_PATH,
    output_dir=CHUNKED_PATH,
    chunk_size=600,
    maintain_folder_structure=True
)
chunker.run_pipeline()

# Step 5: Create case-based vectorstores
print("\n=== Step 5: Case-Based Vectorization ===")

# Option A: Vectorize all cases automatically (if your chunked data is organized in case subdirectories)
print("\n--- Option A: Auto-detect and vectorize all cases ---")
rag_auto = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE,
    source_type_configs=SOURCE_TYPE_CONFIGS,
    consolidation_configs=CONSOLIDATION_CONFIGS,
    chunked_path=CHUNKED_PATH,
    vectorstore_path=VECTORSTORE_PATH
)

# Check what cases are available
available_cases = rag_auto.get_available_cases()
print(f"Available cases detected: {available_cases}")

# Vectorize all detected cases
if available_cases:
    rag_auto.vectorize_all_cases(embeddings_type=VECTORSTORE_TYPE)
else:
    print("No case subdirectories found. Creating default vectorstore...")
    rag_auto.vectorize_documents(embeddings_type=VECTORSTORE_TYPE)

# Step 6: List all created vectorstores
print("\n=== Step 6: Available Vectorstores ===")
rag_auto.list_available_vectorstores()

# Step 7: Example of running pipeline for a specific case (NSCLC)
print("\n=== Step 7: Running Pipeline for NSCLC Case ===")

# Initialize case-specific RAG pipeline
rag_nsclc = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE,
    case="NSCLC",
    source_type_configs=SOURCE_TYPE_CONFIGS,
    consolidation_configs=CONSOLIDATION_CONFIGS,
    chunked_path=CHUNKED_PATH,
    vectorstore_path=VECTORSTORE_PATH
)

# Load the NSCLC vectorstore
rag_nsclc.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE, case="NSCLC")

# Get NSCLC case configuration
nsclc_config = CASE_CONFIGS["case_1_nsclc_krasg12c_monotherapy_progressed"]
nsclc_indication = nsclc_config["indication"]
nsclc_required_terms = nsclc_config.get("required_terms_clinical")
nsclc_mutation_boost = nsclc_config.get("mutation_boost_terms", [])

# Run retrieval for NSCLC case
print("\n--- Running NSCLC Retrieval ---")

# Run HTA Population & Comparator retrieval
print("Running HTA Population & Comparator retrieval for NSCLC...")
rag_nsclc.run_population_comparator_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=nsclc_indication,
    mutation_boost_terms=nsclc_mutation_boost,
    initial_k=60,
    final_k=22
)

# Run HTA Outcomes retrieval
print("Running HTA Outcomes retrieval for NSCLC...")
rag_nsclc.run_outcomes_retrieval_for_source_type(
    source_type="hta_submission",
    countries=COUNTRIES,
    indication=nsclc_indication,
    mutation_boost_terms=nsclc_mutation_boost,
    initial_k=60,
    final_k=22
)

# Run Clinical Guideline retrievals
print("Running Clinical Guideline retrievals for NSCLC...")
rag_nsclc.run_population_comparator_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=nsclc_indication,
    required_terms=nsclc_required_terms,
    mutation_boost_terms=nsclc_mutation_boost,
    initial_k=60,
    final_k=12
)

rag_nsclc.run_outcomes_retrieval_for_source_type(
    source_type="clinical_guideline",
    countries=COUNTRIES,
    indication=nsclc_indication,
    required_terms=nsclc_required_terms,
    mutation_boost_terms=nsclc_mutation_boost,
    initial_k=60,
    final_k=12
)

# Initialize PICO extractors and run extractions
print("\n--- Running NSCLC PICO Extraction ---")
rag_nsclc.initialize_pico_extractors()

# Extract PICOs from HTA submissions
print("Extracting NSCLC HTA Submission PICOs...")
extracted_picos_hta_nsclc = rag_nsclc.run_pico_extraction_for_source_type(
    source_type="hta_submission",
    indication=nsclc_indication
)

# Extract PICOs from clinical guidelines
print("Extracting NSCLC Clinical Guideline PICOs...")
extracted_picos_clinical_nsclc = rag_nsclc.run_pico_extraction_for_source_type(
    source_type="clinical_guideline",
    indication=nsclc_indication
)

# Run consolidation
print("\n--- Running NSCLC PICO Consolidation ---")
rag_nsclc.initialize_pico_consolidator()
consolidation_results_nsclc = rag_nsclc.run_pico_consolidation(
    source_types=["hta_submission", "clinical_guideline"]
)

# Step 8: Example of running pipeline for HCC case (if configured)
print("\n=== Step 8: Running Pipeline for HCC Case (Example) ===")

if "HCC" in available_cases:
    # Initialize HCC-specific RAG pipeline
    rag_hcc = RagPipeline(
        model=MODEL,
        vectorstore_type=VECTORSTORE_TYPE,
        case="HCC",
        source_type_configs=SOURCE_TYPE_CONFIGS,
        consolidation_configs=CONSOLIDATION_CONFIGS,
        chunked_path=CHUNKED_PATH,
        vectorstore_path=VECTORSTORE_PATH
    )
    
    # Load the HCC vectorstore
    rag_hcc.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE, case="HCC")
    
    # For HCC, you would define similar case configuration
    # This is just an example - you'd need to create HCC_CASE_CONFIG
    hcc_indication = "hepatocellular carcinoma"  # Example indication
    
    print("Running HCC retrieval and extraction pipeline...")
    # [Similar retrieval and extraction steps for HCC]
    
else:
    print("HCC case not found in available cases. Skipping HCC pipeline.")

# Step 9: Cross-case analysis (optional)
print("\n=== Step 9: Cross-Case Analysis ===")

# Example: Compare vectorstores from different cases
print("Comparing vectorstores across cases...")

for case in available_cases:
    print(f"\n--- Case: {case} ---")
    vectorstore = rag_auto.load_vectorstore_for_case(case, VECTORSTORE_TYPE)
    if vectorstore:
        # Get some statistics
        result = vectorstore.get(limit=5, include=['metadatas'])
        print(f"Sample documents: {len(result['ids'])}")
        if result['metadatas']:
            countries = set(md.get('country', 'unknown') for md in result['metadatas'])
            source_types = set(md.get('source_type', 'unknown') for md in result['metadatas'])
            print(f"Countries: {countries}")
            print(f"Source types: {source_types}")

# Step 10: Analysis and Visualization (for NSCLC as example)
print("\n=== Step 10: Analysis and Visualization ===")

# Find the most recent consolidated files for NSCLC
nsclc_consolidated_dir = Path("results/NSCLC/consolidated")
if nsclc_consolidated_dir.exists():
    # Get the most recent PICO and Outcomes files
    pico_files = list(nsclc_consolidated_dir.glob("*consolidated_picos*.json"))
    outcome_files = list(nsclc_consolidated_dir.glob("*consolidated_outcomes*.json"))
    
    if pico_files and outcome_files:
        # Sort by modification time and get the most recent
        pico_file = max(pico_files, key=os.path.getmtime)
        outcome_file = max(outcome_files, key=os.path.getmtime)
        
        print(f"Analyzing NSCLC PICO data from: {pico_file}")
        print(f"Analyzing NSCLC Outcomes data from: {outcome_file}")
        
        # Run comprehensive analysis for NSCLC
        try:
            pico_analyzer, outcome_analyzer, visualizer = run_complete_analysis(
                pico_file_path=str(pico_file),
                outcome_file_path=str(outcome_file)
            )
            print("NSCLC analysis completed successfully!")
        except Exception as e:
            print(f"Error in NSCLC analysis: {e}")
    else:
        print("Warning: Could not find NSCLC consolidated files.")
else:
    print("Warning: NSCLC consolidated directory not found.")

print("\n" + "="*80)
print("CASE-BASED RAG PIPELINE EXECUTION COMPLETE")
print("="*80)
print("Summary:")
print(f"- Available cases: {', '.join(available_cases) if available_cases else 'None detected'}")
print(f"- Vectorstore type: {VECTORSTORE_TYPE}")
print(f"- Results organized by case in results/ directory")
print("- Check individual case directories for analysis results")
print("="*80)