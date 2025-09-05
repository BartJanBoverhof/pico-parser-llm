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

# Summary
print("\n=== RETRIEVAL, EXTRACTION & CONSOLIDATION PIPELINE EXECUTION SUMMARY ===")
print("✓ Documents processed and vectorized")
print("✓ Retrieval system initialized with specialized capabilities")
print("✓ Population & Comparator chunk retrieval completed and saved")
print("✓ Outcomes chunk retrieval completed and saved")
print("✓ PICO extraction completed:")
print("  - Population & Comparator extracted separately")
print("  - Outcomes extracted separately")
print("  - Results combined into final PICO format")
print(f"✓ Case 1 HTA submissions: {len(extracted_picos_hta_case1) if extracted_picos_hta_case1 else 0} countries processed")
print(f"✓ Case 1 Clinical guidelines: {len(extracted_picos_clinical_case1) if extracted_picos_clinical_case1 else 0} countries processed")

# Consolidation summary
if consolidation_results and "summary" in consolidation_results:
    summary = consolidation_results["summary"]
    print("✓ PICO and Outcomes consolidation completed:")
    print(f"  - Original PICOs: {summary.get('original_picos', 0)}")
    print(f"  - Consolidated PICOs: {summary.get('consolidated_picos', 0)}")
    print(f"  - Original outcomes: {summary.get('original_outcomes', 0)}")
    print(f"  - Categorized unique outcomes: {summary.get('unique_outcomes', 0)}")
    print(f"  - Countries included: {', '.join(summary.get('countries', []))}")
    print(f"  - Source types: {', '.join(summary.get('source_types', []))}")

print(f"✓ Model used: {MODEL}")
print(f"✓ Vectorstore: {VECTORSTORE_TYPE}")
print("✓ Complete pipeline with consolidation successfully implemented")

# Print file locations for retrieval, extraction, and consolidation
print("\n=== PIPELINE OUTPUT FILES ===")
print("📁 Retrieval Results:")
print("  - HTA Population & Comparator chunks: results/chunks/hta_submission_population_comparator_*_retrieval_results.json")
print("  - HTA Outcomes chunks: results/chunks/hta_submission_outcomes_*_retrieval_results.json")
print("  - Clinical Guideline Population & Comparator chunks: results/chunks/clinical_guideline_population_comparator_*_retrieval_results.json")
print("  - Clinical Guideline Outcomes chunks: results/chunks/clinical_guideline_outcomes_*_retrieval_results.json")
print("📁 Extraction Results:")
print("  - HTA submission PICOs (combined): results/PICO/hta_submission_picos.json")
print("  - Clinical guideline PICOs (combined): results/PICO/clinical_guideline_picos.json")
print("  - Separate outcomes extractions: results/chunks/*_outcomes_*_extraction_results.json")
print("📁 Consolidation Results:")
print("  - Consolidated PICOs & Outcomes: results/consolidated/*_consolidated_*.json")

# Print extraction advantages
print("\n=== EXTRACTION ADVANTAGES ===")
print("🔄 Specialized Prompting:")
print("  - Population & Comparator extraction uses focused prompts")
print("  - Outcomes extraction uses dedicated prompts")
print("  - Better separation of concerns leads to more accurate extraction")
print("🔄 Improved Accuracy:")
print("  - Each extraction step focuses on specific PICO elements")
print("  - Reduces confusion between different types of information")
print("  - Maintains same final JSON structure as before")
print("🔄 Enhanced Flexibility:")
print("  - Can adjust retrieval parameters separately for P&C vs O")
print("  - Can fine-tune extraction prompts independently")
print("  - Backward compatible with existing workflows")

# Print consolidation advantages
print("\n=== CONSOLIDATION ADVANTAGES ===")
print("🔄 Cross-Country Harmonization:")
print("  - Merges similar PICOs from different countries while preserving outcomes")
print("  - Combines outcomes from consolidated PICOs into comprehensive descriptions")
print("  - Tracks country and source type origins")
print("  - Reduces redundancy while preserving important variations")
print("🔄 Structured Outcomes Organization:")
print("  - Categorizes outcomes by clinical relevance (Efficacy, Safety, QoL, Economic)")
print("  - Removes duplicates while preserving measurement details")
print("  - Creates organized reference for outcome planning")
print("🔄 Enhanced Analysis Ready:")
print("  - Consolidated PICOs include comprehensive outcomes information")
print("  - Structured outcomes support systematic review")
print("  - Maintains traceability to original sources")

print("\n=== EXTRACTION METHODOLOGY ===")
print("1. Population & Comparator Extraction:")
print("   - Uses population_comparator chunks from retrieval")
print("   - Specialized prompts focus on patient definitions and treatment alternatives")
print("   - Creates PICO entries with empty Outcomes field")
print("2. Outcomes Extraction:")
print("   - Uses outcomes chunks from retrieval")
print("   - Specialized prompts focus on clinical endpoints and safety")
print("   - Extracts consolidated outcomes per country")
print("3. Results Combination:")
print("   - Merges Population & Comparator entries with Outcomes")
print("   - Maintains original PICO JSON structure")
print("   - Each PICO entry gets the country-specific outcomes")
print("4. PICO Consolidation:")
print("   - LLM-based consolidation of similar PICOs across countries")
print("   - Combines outcomes from consolidated PICOs into comprehensive descriptions")
print("   - Preserves clinical distinctions while reducing redundancy")
print("   - Tracks origins and maintains traceability")
print("5. Outcomes Consolidation:")
print("   - Categorizes outcomes into clinical domains")
print("   - Organizes by relevance and measurement approach")
print("   - Creates structured outcome reference")

print("\n=== NEXT STEPS ===")
print("1. Review the extraction results to validate separation quality")
print("2. Fine-tune extraction prompts based on results")
print("3. Adjust retrieval parameters for optimal P&C vs Outcomes separation")
print("4. Review consolidation results for accuracy and completeness")
print("5. Verify that consolidated PICOs include comprehensive outcomes information")
print("6. Fine-tune consolidation prompts based on domain expertise")
print("7. Test with additional case configurations")
print("8. Validate that final consolidated structures meet analysis needs")

# Final validation
if extracted_picos_hta_case1:
    print(f"\n✅ Extraction successful!")
    total_hta_picos = 0
    if isinstance(extracted_picos_hta_case1, dict) and "picos_by_country" in extracted_picos_hta_case1:
        total_hta_picos = sum(len(country.get('PICOs', [])) for country in extracted_picos_hta_case1["picos_by_country"].values())
    print(f"📊 HTA Results: {total_hta_picos} total PICOs extracted")
    
    if extracted_picos_clinical_case1:
        total_clinical_picos = 0
        if isinstance(extracted_picos_clinical_case1, dict) and "picos_by_country" in extracted_picos_clinical_case1:
            total_clinical_picos = sum(len(country.get('PICOs', [])) for country in extracted_picos_clinical_case1["picos_by_country"].values())
        print(f"📊 Clinical Results: {total_clinical_picos} total PICOs extracted")
    
    if consolidation_results and "summary" in consolidation_results:
        summary = consolidation_results["summary"]
        print(f"🎯 Consolidation Results: {summary.get('consolidated_picos', 0)} consolidated PICOs with outcomes, {summary.get('unique_outcomes', 0)} categorized outcomes")
else:
    print("❌ Extraction failed - check logs for details")