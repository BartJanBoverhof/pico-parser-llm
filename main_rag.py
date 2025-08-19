# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key

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
# A. Initialize RAG system with specialized retrieval methods
rag = RagPipeline(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE
)

# Load the vectorstore
rag.vectorize_documents(embeddings_type=VECTORSTORE_TYPE)

# Initialize the retriever with the created vectorstore
rag.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE)

# Test 1: Test HTA submission retrieval with specialized method
print("\n--- Testing HTA Submission Retrieval (Specialized) ---")
hta_test_results = rag.test_retrieval(
    query=rag.default_query_hta,
    countries=COUNTRIES,
    source_type="hta_submission",
    heading_keywords=[
        "comparator", "alternative", "treatment", "therapy", "intervention",
        "population", "outcomes", "efficacy", "safety", "pico",
        "appropriate comparator therapy", "designation of therapy",
        "medicinal product", "clinical trial"
    ],
    drug_keywords=["docetaxel", "nintedanib", "pembrolizumab", "sotorasib", "adagrasib", "lenvatinib", "sorafenib"],
    initial_k=30,
    final_k=15
)

# Test 2: Test Clinical Guideline retrieval with strict KRAS G12C filtering
print("\n--- Testing Clinical Guideline Retrieval (Strict KRAS G12C) ---")
clinical_test_results = rag.test_retrieval(
    query=rag.default_query_clinical,
    countries=COUNTRIES,
    source_type="clinical_guideline",
    heading_keywords=[
        "recommendation", "treatment", "therapy", "algorithm", "guideline",
        "kras", "g12c", "mutation", "nsclc", "lung cancer",
        "second line", "progression", "targeted therapy"
    ],
    drug_keywords=["sotorasib", "adagrasib", "kras", "g12c"],
    initial_k=50,
    final_k=10
)

# Test 3: Test general retrieval (no source type filter) - fallback method
print("\n--- Testing General Retrieval (All Source Types) ---")
general_test_results = rag.test_retrieval(
    query="KRAS G12C mutation advanced NSCLC treatment comparators",
    countries=COUNTRIES,
    source_type=None,  # No filter - get from all sources
    heading_keywords=["treatment", "therapy", "comparator", "kras", "g12c"],
    drug_keywords=["sotorasib", "adagrasib", "docetaxel", "pembrolizumab"],
    initial_k=20,
    final_k=10
)

print("\n=== SPECIALIZED RETRIEVAL TESTING COMPLETE ===\n")

# Initialize separate PICO extractors for HTA submissions and clinical guidelines
rag.initialize_pico_extractors()

# Process HTA submissions with specialized retrieval and enhanced query
print("\n--- Extracting HTA Submission PICOs with Specialized Retrieval ---")
extracted_picos_hta = rag.extract_picos_hta(countries=COUNTRIES)

# Process clinical guidelines with strict KRAS G12C filtering
print("\n--- Extracting Clinical Guideline PICOs with Strict KRAS G12C Filtering ---")
extracted_picos_clinical = rag.extract_picos_clinical(countries=COUNTRIES)

# Print extracted PICOs with specialized retrieval insights
print("\n=== HTA SUBMISSION PICOS (Specialized Retrieval) ===")
for pico in extracted_picos_hta:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Retrieval Method: HTA-specialized (PICO/comparator focus)")
    if pico_count > 0:
        # Show first PICO as example
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Comparator: {first_pico.get('Comparator', 'N/A')}")
    print("---")

print("\n=== CLINICAL GUIDELINE PICOS (Strict KRAS G12C Filtering) ===")
for pico in extracted_picos_clinical:
    country = pico.get('Country', 'Unknown')
    pico_count = len(pico.get('PICOs', []))
    print(f"Country: {country}")
    print(f"Number of PICOs: {pico_count}")
    print(f"Retrieval Method: Clinical guidelines with strict KRAS G12C filtering")
    if pico_count > 0:
        # Show first PICO as example
        first_pico = pico.get('PICOs', [{}])[0]
        print(f"Sample PICO - Population: {first_pico.get('Population', 'N/A')[:100]}...")
        print(f"Sample PICO - Intervention: {first_pico.get('Intervention', 'N/A')}")
    print("---")

# Summary of specialized retrieval approach
print("\n=== SPECIALIZED RETRIEVAL SUMMARY ===")
print("✓ HTA Submissions:")
print("  - Leverages structured nature of submissions")
print("  - Prioritizes PICO elements and comparator sections")
print("  - Boosts chunks with treatment/intervention keywords")
print("  - Optimizes for comprehensive comparator coverage")
print("")
print("✓ Clinical Guidelines:")
print("  - Strict KRAS G12C mutation filtering")
print("  - Requires explicit mention of both KRAS G12C and NSCLC")
print("  - Focuses on mutation-specific recommendations")
print("  - Prioritizes post-progression therapy content")
print("")
print("✓ Both approaches:")
print("  - Retrieve chunks on per-document/country basis")
print("  - Use dedicated deduplication and context optimization")
print("  - Maintain separate processing pipelines")
print("  - Apply source-specific scoring and ranking")