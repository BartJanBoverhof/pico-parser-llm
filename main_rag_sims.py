from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline, SimulationRunner
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS, CONSOLIDATION_CONFIGS, SIMULATION_CONFIGS
from python.results import RunResults
import glob
import os
from pathlib import Path

PDF_PATH = "data/PDF"
CLEAN_PATH = "data/text_cleaned"
TRANSLATED_PATH = "data/text_translated"
POST_CLEANED_PATH = "data/post_cleaned"
CHUNKED_PATH_BASE = "data/text_chunked"
VECTORSTORE_PATH_BASE = "data/vectorstore"
RESULTS_PATH_BASE = "results"
VECTORSTORE_TYPE = "biobert"
MODEL = "gpt-4.1"
COUNTRIES = ["ALL"]
CASES = ["nsclc", "hcc"]


def run_simulation_pipeline(
    sim_runner,
    chunked_path,
    vectorstore_path,
    results_path,
    chunk_params,
    extraction_temperature,
    pdf_path=PDF_PATH,
    clean_path=CLEAN_PATH,
    translated_path=TRANSLATED_PATH,
    post_cleaned_path=POST_CLEANED_PATH,
    vectorstore_type=VECTORSTORE_TYPE,
    model=MODEL,
    countries=COUNTRIES,
    cases=CASES
):
    tree = FolderTree(root_path=".")
    tree.generate()
    
    if sim_runner._needs_new_vectorstore():
        print("\n" + "="*80)
        print("SIMULATION-SPECIFIC PREPROCESSING")
        print("="*80)
        
        print("\n=== Step 4: Chunking Documents ===")
        chunker = Chunker(
            json_folder_path=post_cleaned_path,
            output_dir=chunked_path,
            chunk_size=chunk_params["min_chunk_size"],
            maintain_folder_structure=True,
            max_chunk_size=chunk_params["max_chunk_size"],
            min_chunk_size=chunk_params["min_chunk_size"]
        )
        chunker.run_pipeline()
        
        print("\n" + "="*80)
        print("SIMULATION-SPECIFIC PREPROCESSING COMPLETED")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("REUSING BASELINE PREPROCESSING")
        print("="*80)
        print("This simulation uses the same chunk parameters as baseline.")
        print("Skipping PDF processing, translation, and chunking steps.")
        print("="*80)
    
    hta_pc_params = sim_runner.get_retrieval_params("hta_submission", "population_comparator")
    hta_o_params = sim_runner.get_retrieval_params("hta_submission", "outcomes")
    clinical_pc_params = sim_runner.get_retrieval_params("clinical_guideline", "population_comparator")
    clinical_o_params = sim_runner.get_retrieval_params("clinical_guideline", "outcomes")
    
    for case_name in cases:
        print("\n" + "="*80)
        print(f"PROCESSING CASE: {case_name.upper()}")
        print("="*80)
        
        print(f"\n=== Step 5: Creating {case_name.upper()} Vectorstore ===")
        vectoriser = Vectoriser(
            chunked_folder_path=chunked_path,
            embedding_choice=vectorstore_type,
            db_parent_dir=vectorstore_path,
            case=case_name
        )
        vectorstore = vectoriser.run_pipeline()
        
        print(f"\n=== Step 6: Initializing {case_name.upper()} RAG System ===")
        rag = RagPipeline(
            model=model,
            vectorstore_type=vectorstore_type,
            case=case_name,
            source_type_configs=SOURCE_TYPE_CONFIGS,
            consolidation_configs=CONSOLIDATION_CONFIGS,
            chunked_path=chunked_path,
            vectorstore_path=vectorstore_path,
            results_path=results_path
        )
        
        rag.vectorize_documents(embeddings_type=vectorstore_type, case=case_name)
        rag.initialize_retriever(vectorstore_type=vectorstore_type, case=case_name)
        
        print(f"\n=== Step 7: Loading {case_name.upper()} Case Configuration ===")
        case_config_key = f"case_1_{case_name}_krasg12c_monotherapy_progressed" if case_name == "nsclc" else f"case_2_{case_name}_advanced_unresectable"
        case_config = CASE_CONFIGS[case_config_key]
        case_indication = case_config["indication"]
        case_required_terms = case_config.get("required_terms_clinical")
        case_mutation_boost = case_config.get("mutation_boost_terms", [])
        
        print(f"\n=== Step 8: Running {case_name.upper()} Retrieval ===")
        
        print(f"\n--- Running HTA Population & Comparator retrieval for {case_name.upper()} ---")
        result_hta_pc = rag.run_population_comparator_retrieval_for_source_type(
            source_type="hta_submission",
            countries=countries,
            indication=case_indication,
            mutation_boost_terms=case_mutation_boost,
            initial_k=hta_pc_params["initial_k"],
            final_k=hta_pc_params["final_k"]
        )
        
        print(f"\n--- Running HTA Outcomes retrieval for {case_name.upper()} ---")
        result_hta_outcomes = rag.run_outcomes_retrieval_for_source_type(
            source_type="hta_submission",
            countries=countries,
            indication=case_indication,
            mutation_boost_terms=case_mutation_boost,
            initial_k=hta_o_params["initial_k"],
            final_k=hta_o_params["final_k"]
        )
        
        print(f"\n--- Running Clinical Guideline Population & Comparator retrieval for {case_name.upper()} ---")
        result_clinical_pc = rag.run_population_comparator_retrieval_for_source_type(
            source_type="clinical_guideline",
            countries=countries,
            indication=case_indication,
            required_terms=case_required_terms,
            mutation_boost_terms=case_mutation_boost,
            initial_k=clinical_pc_params["initial_k"],
            final_k=clinical_pc_params["final_k"]
        )
        
        print(f"\n--- Running Clinical Guideline Outcomes retrieval for {case_name.upper()} ---")
        result_clinical_outcomes = rag.run_outcomes_retrieval_for_source_type(
            source_type="clinical_guideline",
            countries=countries,
            indication=case_indication,
            required_terms=case_required_terms,
            mutation_boost_terms=case_mutation_boost,
            initial_k=clinical_o_params["initial_k"],
            final_k=clinical_o_params["final_k"]
        )
        
        print(f"\n=== Step 9: Initializing {case_name.upper()} PICO Extractors ===")
        rag.initialize_pico_extractors(temperature=extraction_temperature)
        
        print(f"\n=== Step 10: Running {case_name.upper()} PICO Extraction ===")
        
        print(f"\n--- Extracting {case_name.upper()} HTA Submission PICOs ---")
        extracted_picos_hta = rag.run_pico_extraction_for_source_type(
            source_type="hta_submission",
            indication=case_indication
        )
        
        print(f"\n--- Extracting {case_name.upper()} HTA Submission Outcomes ---")
        extracted_outcomes_hta = rag.run_outcomes_extraction_for_source_type(
            source_type="hta_submission",
            indication=case_indication
        )
        
        print(f"\n--- Extracting {case_name.upper()} Clinical Guideline PICOs ---")
        extracted_picos_clinical = rag.run_pico_extraction_for_source_type(
            source_type="clinical_guideline",
            indication=case_indication
        )
        
        print(f"\n--- Extracting {case_name.upper()} Clinical Guideline Outcomes ---")
        extracted_outcomes_clinical = rag.run_outcomes_extraction_for_source_type(
            source_type="clinical_guideline",
            indication=case_indication
        )
        
        print(f"\n=== Step 11: Running {case_name.upper()} PICO and Outcomes Consolidation ===")
        
        rag.initialize_pico_consolidator()
        
        print(f"\n--- Consolidating {case_name.upper()} PICOs and Outcomes for Test Set ---")
        consolidation_results_test = rag.run_pico_consolidation(
            source_types=["hta_submission", "clinical_guideline"],
            test_set=True
        )
    
    print("\n" + "="*80)
    print(f"SIMULATION {sim_runner.simulation_id} COMPLETED")
    print("="*80)

validate_api_key()

simulation_ids = ["base_b, base_c,", "base_d", "base_e"]

for simulation_id in simulation_ids:
    print("\n" + "="*80)
    print(f"INITIALIZING SIMULATION: {simulation_id}")
    print("="*80)
    
    sim_runner = SimulationRunner(
        simulation_id=simulation_id,
        base_paths={
            "chunked": CHUNKED_PATH_BASE,
            "vectorstore": VECTORSTORE_PATH_BASE,
            "results": RESULTS_PATH_BASE
        },
        case=None
    )
    
    sim_runner.print_summary()
    sim_runner.create_folders()
    
    sim_paths = sim_runner.get_paths()
    chunk_params = sim_runner.get_chunk_params()
    extraction_temperature = sim_runner.get_extraction_temperature()
    
    chunked_path = sim_paths["chunked"]
    vectorstore_path = sim_paths["vectorstore"]
    results_path = sim_paths["results"]
    
    run_simulation_pipeline(
        sim_runner=sim_runner,
        chunked_path=chunked_path,
        vectorstore_path=vectorstore_path,
        results_path=results_path,
        chunk_params=chunk_params,
        extraction_temperature=extraction_temperature
    )
    
    print("\n" + "#"*80)
    print(f"# SIMULATION {simulation_id} FINISHED")
    print("#"*80 + "\n")

print("\n" + "="*80)
print("ALL SIMULATIONS COMPLETED")
print("="*80)
print(f"Completed {len(simulation_ids)} simulations: {', '.join(simulation_ids)}")
print("="*80)

print("\n" + "="*80)
print("GENERATING CONSOLIDATED RESULTS ACROSS ALL SIMULATIONS")
print("="*80)

results_runner = RunResults(
    translated_path=TRANSLATED_PATH,
    results_path=RESULTS_PATH_BASE,
    mode="consolidated_only"
)

results_runner.run_all()