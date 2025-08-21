# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, TEST_QUERIES, CASE_CONFIGS

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
print("\n--- Testing Case 1: NSCLC KRAS G12C Retrieval ---")
case1_indication = CASE_CONFIGS["case_1_nsclc_krasg12c_monotherapy_progressed"]["indication"]

# Test HTA submission retrieval for Case 1
hta_config = SOURCE_TYPE_CONFIGS["hta_submission"]
hta_query_case1 = hta_config["query_template"].format(indication=case1_indication)

hta_test_results_case1 = rag.test_retrieval(
    query=hta_query_case1,
    countries=COUNTRIES,
    source_type="hta_submission",
    heading_keywords=hta_config["default_headings"],
    drug_keywords=hta_config["default_drugs"],
    initial_k=24,
    final_k=12
)

# Test Clinical Guideline retrieval for Case 1
clinical_config = SOURCE_TYPE_CONFIGS["clinical_guideline"]
clinical_query_case1 = clinical_config["query_template"].format(indication=case1_indication)

clinical_test_results_case1 = rag.test_retrieval(
    query=clinical_query_case1,
    countries=COUNTRIES,
    source_type="clinical_guideline",
    heading_keywords=clinical_config["default_headings"],
    drug_keywords=clinical_config["default_drugs"],
    initial_k=60,
    final_k=12
)

# Initialize PICO extractors for both source types
rag.initialize_pico_extractors()

# Process HTA submissions for Case 1
print("\n--- Extracting Case 1 HTA Submission PICOs ---")
extracted_picos_hta_case1 = rag.extract_picos_hta_with_indication(
    countries=COUNTRIES,
    indication=case1_indication
)

# Process clinical guidelines for Case 1
print("\n--- Extracting Case 1 Clinical Guideline PICOs ---")
extracted_picos_clinical_case1 = rag.extract_picos_clinical_with_indication(
    countries=COUNTRIES,
    indication=case1_indication
)

# Print extracted PICOs for Case 1
print("\n=== CASE 1 - NSCLC KRAS G12C HTA SUBMISSION PICOS ===")
for pico in extracted_picos_hta_case1:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    chunks_used = pico.get('ChunksUsed', 0)
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Chunks used: {chunks_used}")
    if pico_count > 0:
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Intervention: {first_pico.get('Intervention', 'N/A')[:50]}...")
        print(f"Sample PICO - Comparator: {first_pico.get('Comparator', 'N/A')[:50]}...")
    print("---")

print("\n=== CASE 1 - NSCLC KRAS G12C CLINICAL GUIDELINE PICOS ===")
for pico in extracted_picos_clinical_case1:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    chunks_used = pico.get('ChunksUsed', 0)
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Chunks used: {chunks_used}")
    if pico_count > 0:
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Intervention: {first_pico.get('Intervention', 'N/A')[:50]}...")
        print(f"Sample PICO - Comparator: {first_pico.get('Comparator', 'N/A')[:50]}...")
    print("---")

"""
# Test Case 2: Hepatocellular Carcinoma
print("\n--- Testing Case 2: HCC Advanced Unresectable Retrieval ---")
case2_indication = CASE_CONFIGS["case_2_hcc_advanced_unresectable"]["indication"]

# Test HTA submission retrieval for Case 2
hta_query_case2 = hta_config["query_template"].format(indication=case2_indication)

hta_test_results_case2 = rag.test_retrieval(
    query=hta_query_case2,
    countries=COUNTRIES,
    source_type="hta_submission",
    heading_keywords=hta_config["default_headings"],
    drug_keywords=hta_config["default_drugs"],
    initial_k=30,
    final_k=15
)

# Test Clinical Guideline retrieval for Case 2
clinical_query_case2 = clinical_config["query_template"].format(indication=case2_indication)

clinical_test_results_case2 = rag.test_retrieval(
    query=clinical_query_case2,
    countries=COUNTRIES,
    source_type="clinical_guideline",
    heading_keywords=clinical_config["default_headings"],
    drug_keywords=clinical_config["default_drugs"],
    initial_k=50,
    final_k=10
)


# Process HTA submissions for Case 2
print("\n--- Extracting Case 2 HTA Submission PICOs ---")
extracted_picos_hta_case2 = rag.extract_picos_hta_with_indication(
    countries=COUNTRIES,
    indication=case2_indication
)


# Process clinical guidelines for Case 2
print("\n--- Extracting Case 2 Clinical Guideline PICOs ---")
extracted_picos_clinical_case2 = rag.extract_picos_clinical_with_indication(
    countries=COUNTRIES,
    indication=case2_indication
)

# Print extracted PICOs for Case 2
print("\n=== CASE 2 - HCC ADVANCED UNRESECTABLE HTA SUBMISSION PICOS ===")
for pico in extracted_picos_hta_case2:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    chunks_used = pico.get('ChunksUsed', 0)
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Chunks used: {chunks_used}")
    if pico_count > 0:
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Intervention: {first_pico.get('Intervention', 'N/A')[:50]}...")
        print(f"Sample PICO - Comparator: {first_pico.get('Comparator', 'N/A')[:50]}...")
    print("---")

print("\n=== CASE 2 - HCC ADVANCED UNRESECTABLE CLINICAL GUIDELINE PICOS ===")
for pico in extracted_picos_clinical_case2:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    chunks_used = pico.get('ChunksUsed', 0)
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Chunks used: {chunks_used}")
    if pico_count > 0:
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Intervention: {first_pico.get('Intervention', 'N/A')[:50]}...")
        print(f"Sample PICO - Comparator: {first_pico.get('Comparator', 'N/A')[:50]}...")
    print("---")
"""

# Summary
print("\n=== PIPELINE EXECUTION SUMMARY ===")
print("✓ Documents processed and vectorized")
print("✓ Retrieval system initialized with enhanced capabilities")
print("✓ PICO extraction completed for both source types and cases")
print(f"✓ Results saved to JSON files in 'results' directory")
print(f"✓ Case 1 HTA submissions: {len(extracted_picos_hta_case1)} countries processed")
print(f"✓ Case 1 Clinical guidelines: {len(extracted_picos_clinical_case1)} countries processed")
print(f"✓ Case 2 HTA submissions: {len(extracted_picos_hta_case2)} countries processed")
print(f"✓ Case 2 Clinical guidelines: {len(extracted_picos_clinical_case2)} countries processed")
print(f"✓ Model used: {MODEL}")
print(f"✓ Vectorstore: {VECTORSTORE_TYPE}")

# Print file locations
print("\n=== OUTPUT FILES ===")
print("📁 Retrieval results: results/*_retrieval_results.json")
print("📁 Organized PICOs: results/*picos_organized.json")
print("📁 Individual country files: results/*_picos_*.json")