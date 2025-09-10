import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

project_dir = os.getcwd()
if project_dir not in sys.path:
    sys.path.append(project_dir)

from python.utils import FolderTree, HeadingPrinter
from python.process import PDFProcessor
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.retrieve import ChunkRetriever
from python.extract import PICOExtractor
from python.consolidate import PICOConsolidator
from python.open_ai import validate_api_key

from openai import OpenAI

from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class RagPipeline:
    """
    A class to manage the entire RAG (Retrieval-Augmented Generation) pipeline:
    1. PDF processing
    2. Translation
    3. Chunking
    4. Vectorization
    5. Retrieval (separate retrieval for Population & Comparator vs Outcomes)
    6. PICO extraction (separate extraction for Population & Comparator vs Outcomes)
    7. PICO and Outcomes consolidation

    Enhanced to support different source types with specialized retrieval strategies
    and configurable filtering parameters including mutation-specific retrieval.
    Uses separate retrieval and extraction pipelines for Population & Comparator vs Outcomes.
    Enhanced to support case-based vectorstores for different medical cases (e.g., NSCLC, HCC).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        pdf_path: str = "data/PDF",
        clean_path: str = "data/text_cleaned",
        translated_path: str = "data/text_translated",
        chunked_path: str = "data/text_chunked",
        vectorstore_path: str = "data/vectorstore",
        results_path: str = "results",
        chunk_size: int = 600,
        chunk_overlap: int = 200,
        chunk_strategy: str = "semantic",
        vectorstore_type: str = "biobert",
        case: Optional[str] = None,
        source_type_configs: Optional[Dict[str, Any]] = None,
        consolidation_configs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG system with customizable parameters.

        Args:
            model: OpenAI model to use
            pdf_path: Path to the directory containing PDFs
            clean_path: Path to store cleaned text
            translated_path: Path to store translated text
            chunked_path: Path to store chunked text
            vectorstore_path: Path to store vector embeddings
            results_path: Path to store retrieval and PICO extraction results
            chunk_size: Size of chunks for splitting documents
            chunk_overlap: Overlap between chunks
            chunk_strategy: Chunking strategy ("semantic" or "recursive")
            vectorstore_type: Type of vectorstore to use ("openai", "biobert", or "both")
            case: Medical case identifier (e.g., "NSCLC", "HCC")
            source_type_configs: Configuration for different source types
            consolidation_configs: Configuration for PICO and outcomes consolidation
        """
        self.model = model
        self.path_pdf = pdf_path
        self.path_clean = clean_path
        self.path_translated = translated_path
        self.path_chunked = chunked_path
        self.path_vectorstore = vectorstore_path
        self.path_results = results_path
        self.path_chunks = os.path.join(results_path, "chunks")
        self.path_pico = os.path.join(results_path, "PICO")
        self.path_consolidated = os.path.join(results_path, "consolidated")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.vectorstore_type = vectorstore_type
        self.case = case

        # Add case to results path if specified
        if self.case:
            self.path_results = os.path.join(results_path, self.case)
            self.path_chunks = os.path.join(self.path_results, "chunks")
            self.path_pico = os.path.join(self.path_results, "PICO")
            self.path_consolidated = os.path.join(self.path_results, "consolidated")

        os.makedirs(self.path_results, exist_ok=True)
        os.makedirs(self.path_chunks, exist_ok=True)
        os.makedirs(self.path_pico, exist_ok=True)
        os.makedirs(self.path_consolidated, exist_ok=True)

        self.openai = OpenAI()

        self.translator = None
        self.chunker = None
        
        # Case-based vectorizers and vectorstores
        self.vectoriser_openai = None
        self.vectoriser_biobert = None
        self.vectorstore_openai = None
        self.vectorstore_biobert = None
        self.retriever = None
        self.pico_extractor_hta = None
        self.pico_extractor_clinical = None
        self.pico_consolidator = None

        self.source_type_configs = source_type_configs or {}
        self.consolidation_configs = consolidation_configs or {}

    @property
    def chunk_retriever(self):
        """Provide access to the retriever for backward compatibility."""
        return self.retriever

    def show_folder_structure(self, root_path: str = ".", show_hidden: bool = False, max_depth: Optional[int] = None):
        """Show the folder structure of the project."""
        tree = FolderTree(root_path=root_path, show_hidden=show_hidden, max_depth=max_depth)
        tree.generate()

    def print_all_headings(self):
        """Print all detected headings in the translated documents."""
        printer = HeadingPrinter()
        printer.print_all_headings()

    def validate_api_key(self):
        """Validate the OpenAI API key."""
        message = validate_api_key()
        print(message)
        return message

    def get_available_cases(self) -> List[str]:
        """
        Get list of available cases based on subdirectories in the chunked folder.
        
        Returns:
            List of case names
        """
        return Vectoriser.get_available_cases(self.path_chunked)

    def process_pdfs(self):
        """Process PDFs to extract cleaned text."""
        os.makedirs(self.path_clean, exist_ok=True)
        
        PDFProcessor.process_pdfs(self.path_pdf, self.path_clean)
        print(f"Processed PDFs from {self.path_pdf} to {self.path_clean}")

    def translate_documents(self):
        """Translate cleaned text to English."""
        os.makedirs(self.path_translated, exist_ok=True)
        
        if self.translator is None:
            self.translator = Translator(self.path_clean, self.path_translated)
        
        self.translator.translate_documents()
        print(f"Translated documents from {self.path_clean} to {self.path_translated}")

    def chunk_documents(self):
        """Chunk translated documents for vectorization."""
        os.makedirs(self.path_chunked, exist_ok=True)
        
        self.chunker = Chunker(
            json_folder_path=self.path_translated,
            output_dir=self.path_chunked,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_strat=self.chunk_strategy,
            maintain_folder_structure=True
        )
        
        self.chunker.run_pipeline()
        print(f"Chunked documents from {self.path_translated} to {self.path_chunked}")

    def vectorize_documents(
        self, 
        embeddings_type: Optional[str] = None,
        case: Optional[str] = None
    ):
        """
        Vectorize chunked documents using specified embedding type and case.
        
        Args:
            embeddings_type: Type of embeddings to use ("openai", "biobert", or "both")
                           If None, uses the value specified in self.vectorstore_type
            case: Case to vectorize. If None, uses self.case
        """
        if embeddings_type is None:
            embeddings_type = self.vectorstore_type
            
        if case is None:
            case = self.case
            
        os.makedirs(self.path_vectorstore, exist_ok=True)
        
        case_info = f" for case '{case}'" if case else ""
        print(f"Creating vectorstore(s){case_info}...")
        
        if embeddings_type.lower() in ["openai", "both"]:
            self.vectoriser_openai = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="openai",
                db_parent_dir=self.path_vectorstore,
                case=case
            )
            self.vectorstore_openai = self.vectoriser_openai.run_pipeline()
            print(f"Created OpenAI vectorstore{case_info}")
            
        if embeddings_type.lower() in ["biobert", "both"]:
            self.vectoriser_biobert = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="biobert",
                db_parent_dir=self.path_vectorstore,
                case=case
            )
            self.vectorstore_biobert = self.vectoriser_biobert.run_pipeline()
            print(f"Created BioBERT vectorstore{case_info}")
            
        if embeddings_type.lower() == "both" and self.vectoriser_openai and self.vectoriser_biobert:
            print(f"Visualizing vectorstore comparison{case_info}")
            self.vectoriser_openai.visualize_vectorstore(self.vectorstore_biobert)

    def vectorize_all_cases(self, embeddings_type: Optional[str] = None):
        """
        Vectorize all available cases.
        
        Args:
            embeddings_type: Type of embeddings to use ("openai", "biobert", or "both")
        """
        available_cases = self.get_available_cases()
        
        if not available_cases:
            print("No cases found in the chunked directory structure.")
            print("Falling back to default vectorization...")
            self.vectorize_documents(embeddings_type=embeddings_type, case=None)
            return
            
        print(f"Found {len(available_cases)} cases: {', '.join(available_cases)}")
        
        for case in available_cases:
            print(f"\n--- Vectorizing case: {case} ---")
            self.vectorize_documents(embeddings_type=embeddings_type, case=case)

    def initialize_retriever(
        self, 
        vectorstore_type: Optional[str] = None,
        case: Optional[str] = None
    ):
        """
        Initialize the retriever with the specified vectorstore and case.
        
        Args:
            vectorstore_type: Type of vectorstore to use ("openai" or "biobert")
                            If None, uses the value specified in self.vectorstore_type
            case: Case to use for retriever. If None, uses self.case
        """
        if vectorstore_type is None:
            vectorstore_type = self.vectorstore_type
            
        if case is None:
            case = self.case
            
        case_info = f" for case '{case}'" if case else ""
        
        # Load the appropriate case-based vectorstore
        if vectorstore_type.lower() == "openai":
            vectoriser = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="openai",
                db_parent_dir=self.path_vectorstore,
                case=case
            )
            if vectoriser.vectorstore_exists():
                self.vectorstore_openai = vectoriser.load_vectorstore()
                self.retriever = ChunkRetriever(vectorstore=self.vectorstore_openai, results_output_dir=self.path_results)
                print(f"Initialized retriever with OpenAI vectorstore{case_info}")
            else:
                print(f"OpenAI vectorstore not found{case_info}. Please run vectorize_documents first.")
                return
                
        elif vectorstore_type.lower() == "biobert":
            vectoriser = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="biobert",
                db_parent_dir=self.path_vectorstore,
                case=case
            )
            if vectoriser.vectorstore_exists():
                self.vectorstore_biobert = vectoriser.load_vectorstore()
                self.retriever = ChunkRetriever(vectorstore=self.vectorstore_biobert, results_output_dir=self.path_results)
                print(f"Initialized retriever with BioBERT vectorstore{case_info}")
            else:
                print(f"BioBERT vectorstore not found{case_info}. Please run vectorize_documents first.")
                return
        else:
            print(f"Invalid vectorstore_type: {vectorstore_type}. Use 'openai' or 'biobert'.")
            return
            
        print(f"Initialized retriever with {vectorstore_type} vectorstore{case_info} and specialized retrieval methods")

    def load_vectorstore_for_case(
        self, 
        case: str, 
        vectorstore_type: Optional[str] = None
    ):
        """
        Load a specific vectorstore for a given case.
        
        Args:
            case: Case identifier
            vectorstore_type: Type of vectorstore ("openai" or "biobert")
            
        Returns:
            Loaded vectorstore or None if not found
        """
        if vectorstore_type is None:
            vectorstore_type = self.vectorstore_type
            
        vectoriser = Vectoriser(
            chunked_folder_path=self.path_chunked,
            embedding_choice=vectorstore_type,
            db_parent_dir=self.path_vectorstore,
            case=case
        )
        
        if vectoriser.vectorstore_exists():
            return vectoriser.load_vectorstore()
        else:
            print(f"Vectorstore for case '{case}' with {vectorstore_type} embeddings not found.")
            return None

    def list_available_vectorstores(self):
        """
        List all available vectorstores (by case and embedding type).
        """
        if not os.path.exists(self.path_vectorstore):
            print("No vectorstore directory found.")
            return
            
        print("Available vectorstores:")
        for item in os.listdir(self.path_vectorstore):
            item_path = os.path.join(self.path_vectorstore, item)
            if os.path.isdir(item_path):
                # Parse vectorstore name to extract case and embedding type
                if "biobert" in item.lower():
                    embedding_type = "BioBERT"
                    case = item.replace("_biobert_vectorstore", "").replace("biobert_vectorstore", "default")
                elif "openai" in item.lower() or "open_ai" in item.lower():
                    embedding_type = "OpenAI"
                    case = item.replace("_open_ai_vectorstore", "").replace("open_ai_vectorstore", "default")
                else:
                    embedding_type = "Unknown"
                    case = item
                    
                print(f"  - Case: {case}, Embedding: {embedding_type}, Path: {item}")

    def initialize_pico_extractors(self):
        """Initialize separate PICO extractors for HTA and clinical guideline sources."""
        if not self.source_type_configs:
            print("Source type configurations not provided. Cannot initialize PICO extractors.")
            return
            
        case_info = f" for case '{self.case}'" if self.case else ""
        
        if "hta_submission" in self.source_type_configs:
            hta_config = self.source_type_configs["hta_submission"]
            self.pico_extractor_hta = PICOExtractor(
                system_prompt=hta_config.get("population_comparator_system_prompt", ""),
                user_prompt_template=hta_config.get("population_comparator_user_prompt_template", ""),
                source_type="hta_submission",
                model_name=self.model,
                results_output_dir=self.path_results,
                source_type_config=hta_config
            )
        
        if "clinical_guideline" in self.source_type_configs:
            clinical_config = self.source_type_configs["clinical_guideline"]
            self.pico_extractor_clinical = PICOExtractor(
                system_prompt=clinical_config.get("population_comparator_system_prompt", ""),
                user_prompt_template=clinical_config.get("population_comparator_user_prompt_template", ""),
                source_type="clinical_guideline",
                model_name=self.model,
                results_output_dir=self.path_results,
                source_type_config=clinical_config
            )
        
        print(f"Initialized specialized PICO extractors for available source types with model {self.model}{case_info}")

    def initialize_pico_consolidator(self):
        """Initialize the PICO and outcomes consolidator."""
        case_info = f" for case '{self.case}'" if self.case else ""
        self.pico_consolidator = PICOConsolidator(
            model_name=self.model,
            results_output_dir=self.path_results,
            consolidation_configs=self.consolidation_configs
        )
        print(f"Initialized PICO consolidator with model {self.model}{case_info}")

    def get_all_countries(self):
        """
        Retrieves all unique country codes available in the vectorstore.
        
        Returns:
            List[str]: List of unique country codes
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return []
        
        try:
            result = self.retriever.chroma_collection.get(
                limit=10000,
                include=['metadatas']
            )
            
            countries = set()
            for metadata in result['metadatas']:
                if metadata and 'country' in metadata and metadata['country'] not in ['unknown', None, '']:
                    countries.add(metadata['country'])
            
            country_list = sorted(list(countries))
            case_info = f" for case '{self.case}'" if self.case else ""
            print(f"Detected {len(country_list)} countries in vectorstore{case_info}: {', '.join(country_list)}")
            return country_list
            
        except Exception as e:
            print(f"Error retrieving countries from vectorstore: {e}")
            return []

    def run_population_comparator_retrieval_for_source_type(
        self,
        source_type: str,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 50,
        final_k: int = 20,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        indication: Optional[str] = None
    ):
        """
        Run Population & Comparator retrieval for a specific source type and save chunks to files.
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return None
        
        # Handle "ALL" countries by automatically detecting available countries
        if any(country == "ALL" for country in countries):
            all_countries = self.get_all_countries()
            if not all_countries:
                print("No countries detected in the vectorstore. Please check your data.")
                return None
            countries = all_countries
        
        if source_type not in self.source_type_configs:
            raise ValueError(f"Unsupported source_type: {source_type}. Available types: {list(self.source_type_configs.keys())}")
            
        config = self.source_type_configs[source_type]
        
        if query is None:
            if indication:
                query = config["population_comparator_query_template"].format(indication=indication)
            else:
                query = config.get("population_comparator_query_template", "")
            
        if heading_keywords is None:
            heading_keywords = config.get("population_comparator_headings", config.get("default_headings", []))
            
        if drug_keywords is None:
            drug_keywords = config["default_drugs"]
            
        if required_terms is None and "required_terms" in config:
            required_terms = config["required_terms"]
            
        chunks_by_country = self.retriever.retrieve_population_comparator_chunks(
            query=query,
            countries=countries,
            source_type=source_type,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            initial_k=initial_k,
            final_k=final_k
        )

        timestamp = datetime.now().isoformat()
        self.retriever.save_retrieval_results(
            results_by_country=chunks_by_country,
            source_type=source_type,
            retrieval_type="population_comparator",
            query=query,
            timestamp=timestamp,
            indication=indication
        )

        summary = {country: len(chunks) for country, chunks in chunks_by_country.items()}
        total_chunks = sum(summary.values())
        
        print(f"Retrieved chunks by country: {summary}")
        
        return {
            "summary": summary,
            "chunks_by_country": chunks_by_country,
            "timestamp": timestamp
        }

    def run_outcomes_retrieval_for_source_type(
        self,
        source_type: str,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 40,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        indication: Optional[str] = None
    ):
        """
        Run Outcomes retrieval for a specific source type and save chunks to files.
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return None
        
        # Handle "ALL" countries by automatically detecting available countries
        if any(country == "ALL" for country in countries):
            all_countries = self.get_all_countries()
            if not all_countries:
                print("No countries detected in the vectorstore. Please check your data.")
                return None
            countries = all_countries
        
        if source_type not in self.source_type_configs:
            raise ValueError(f"Unsupported source_type: {source_type}. Available types: {list(self.source_type_configs.keys())}")
            
        config = self.source_type_configs[source_type]
        
        if query is None:
            if indication:
                query = config["outcomes_query_template"].format(indication=indication)
            else:
                query = config.get("outcomes_query_template", "")
            
        if heading_keywords is None:
            heading_keywords = config.get("outcomes_headings", config.get("default_headings", []))
            
        if drug_keywords is None:
            drug_keywords = config["default_drugs"]
            
        if required_terms is None and "required_terms" in config:
            required_terms = config["required_terms"]
            
        chunks_by_country = self.retriever.retrieve_outcomes_chunks(
            query=query,
            countries=countries,
            source_type=source_type,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            initial_k=initial_k,
            final_k=final_k
        )

        timestamp = datetime.now().isoformat()
        self.retriever.save_retrieval_results(
            results_by_country=chunks_by_country,
            source_type=source_type,
            retrieval_type="outcomes",
            query=query,
            timestamp=timestamp,
            indication=indication
        )

        summary = {country: len(chunks) for country, chunks in chunks_by_country.items()}
        total_chunks = sum(summary.values())
        
        print(f"Retrieved chunks by country: {summary}")

        return {
            "summary": summary,
            "chunks_by_country": chunks_by_country,
            "timestamp": timestamp
        }

    def run_pico_extraction_for_source_type(
        self,
        source_type: str,
        indication: str,
        countries: Optional[List[str]] = None
    ):
        """
        Run PICO extraction for a specific source type using the most recent retrieval results.
        
        Args:
            source_type: Source type to extract PICOs for
            indication: Medical indication
            countries: List of countries (optional, will use available data if not specified)
            
        Returns:
            Dictionary with extraction results
        """
        case_info = f" for case '{self.case}'" if self.case else ""
        
        if source_type == "hta_submission":
            if self.pico_extractor_hta is None:
                print(f"HTA PICO extractor not initialized{case_info}. Please run initialize_pico_extractors first.")
                return None
            extractor = self.pico_extractor_hta
        elif source_type == "clinical_guideline":
            if self.pico_extractor_clinical is None:
                print(f"Clinical guideline PICO extractor not initialized{case_info}. Please run initialize_pico_extractors first.")
                return None
            extractor = self.pico_extractor_clinical
        else:
            print(f"Unsupported source type: {source_type}")
            return None
        
        # Find the most recent retrieval files for this source type
        population_comparator_file = None
        outcomes_file = None
        
        # Look for population/comparator retrieval files
        pattern = f"{source_type}_population_comparator_*_retrieval_results.json"
        import glob
        pc_files = glob.glob(os.path.join(self.path_chunks, pattern))
        if pc_files:
            population_comparator_file = max(pc_files, key=os.path.getmtime)
        
        # Look for outcomes retrieval files
        pattern = f"{source_type}_outcomes_*_retrieval_results.json"
        outcomes_files = glob.glob(os.path.join(self.path_chunks, pattern))
        if outcomes_files:
            outcomes_file = max(outcomes_files, key=os.path.getmtime)
        
        if not population_comparator_file and not outcomes_file:
            print(f"No retrieval result files found for {source_type}{case_info}. Please run retrieval first.")
            return None
            
        print(f"Extracting PICOs from {source_type}{case_info}...")
        if population_comparator_file:
            print(f"Using population/comparator data from: {os.path.basename(population_comparator_file)}")
        if outcomes_file:
            print(f"Using outcomes data from: {os.path.basename(outcomes_file)}")
        
        # Run the extraction
        results = extractor.extract_picos(
            source_type=source_type,
            indication=indication
        )
        
        return results

    def run_pico_consolidation(
        self,
        source_types: List[str],
        indication: Optional[str] = None
    ):
        """
        Run PICO and outcomes consolidation across multiple source types.
        
        Args:
            source_types: List of source types to consolidate
            indication: Medical indication (optional)
            
        Returns:
            Dictionary with consolidation results
        """
        if self.pico_consolidator is None:
            print("PICO consolidator not initialized. Please run initialize_pico_consolidator first.")
            return None
        
        case_info = f" for case '{self.case}'" if self.case else ""
        print(f"Running PICO and outcomes consolidation across {len(source_types)} source types{case_info}...")
        
        # Check if PICO files exist for the requested source types
        existing_source_types = []
        for source_type in source_types:
            pico_file = os.path.join(self.path_pico, f"{source_type}_picos.json")
            if os.path.exists(pico_file):
                existing_source_types.append(source_type)
                print(f"Found PICO file for {source_type}: {os.path.basename(pico_file)}")
            else:
                print(f"Warning: No PICO extraction file found for {source_type} at {pico_file}")
        
        if not existing_source_types:
            print(f"No PICO extraction files found for consolidation{case_info}. Please run PICO extraction first.")
            return None
        
        # Run consolidation using the consolidate_all method
        results = self.pico_consolidator.consolidate_all(
            source_types=existing_source_types
        )
        
        return results

    def run_case_based_pipeline_with_retrieval(
        self,
        case_config: Dict[str, Any],
        countries: List[str],
        case: Optional[str] = None,
        source_types: List[str] = ["hta_submission", "clinical_guideline"],
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        skip_processing: bool = True,
        skip_translation: bool = True,
        vectorstore_type: Optional[str] = None,
        run_consolidation: bool = False
    ):
        """
        Run pipeline using retrieval for a specific case configuration.
        
        Args:
            case_config: Case configuration dictionary with 'indication' key
            countries: List of country codes or ["ALL"]
            case: Case identifier (overrides self.case if provided)
            source_types: List of source types to process
            initial_k_pc: Initial retrieval count for Population & Comparator
            final_k_pc: Final retrieval count for Population & Comparator
            initial_k_outcomes: Initial retrieval count for Outcomes
            final_k_outcomes: Final retrieval count for Outcomes
            skip_processing: Skip PDF processing
            skip_translation: Skip translation
            vectorstore_type: Vectorstore type to use
            run_consolidation: Whether to run PICO and outcomes consolidation
        
        Returns:
            Dictionary with extracted PICOs for each source type and consolidation results
        """
        if case is not None:
            # Temporarily override the case for this pipeline run
            original_case = self.case
            self.case = case
            # Update paths for this case
            self.path_results = os.path.join("results", self.case)
            self.path_chunks = os.path.join(self.path_results, "chunks")
            self.path_pico = os.path.join(self.path_results, "PICO")
            self.path_consolidated = os.path.join(self.path_results, "consolidated")
            
            os.makedirs(self.path_results, exist_ok=True)
            os.makedirs(self.path_chunks, exist_ok=True)
            os.makedirs(self.path_pico, exist_ok=True)
            os.makedirs(self.path_consolidated, exist_ok=True)
        
        try:
            if vectorstore_type is None:
                vectorstore_type = self.vectorstore_type
                
            indication = case_config.get("indication")
            if not indication:
                raise ValueError("Case configuration must contain 'indication' key")
                
            required_terms = case_config.get("required_terms_clinical")
            mutation_boost_terms = case_config.get("mutation_boost_terms", [])
            drug_keywords = case_config.get("drug_keywords", [])
                
            self.validate_api_key()
            
            if not skip_processing:
                self.process_pdfs()
            
            if not skip_translation:
                self.translate_documents()
                
            if not skip_processing or not skip_translation:
                self.chunk_documents()
                self.vectorize_documents(embeddings_type=vectorstore_type)
            
            self.initialize_retriever(vectorstore_type=vectorstore_type if vectorstore_type != "both" else "biobert")
            
            self.initialize_pico_extractors()
            
            results = {}
            for source_type in source_types:
                if source_type == "hta_submission":
                    extracted_picos = self.extract_picos_hta_with_retrieval(
                        countries=countries,
                        indication=indication,
                        initial_k_pc=initial_k_pc,
                        final_k_pc=final_k_pc,
                        initial_k_outcomes=initial_k_outcomes,
                        final_k_outcomes=final_k_outcomes,
                        drug_keywords=drug_keywords,
                        mutation_boost_terms=mutation_boost_terms
                    )
                elif source_type == "clinical_guideline":
                    extracted_picos = self.extract_picos_clinical_with_retrieval(
                        countries=countries,
                        indication=indication,
                        initial_k_pc=initial_k_pc,
                        final_k_pc=final_k_pc,
                        initial_k_outcomes=initial_k_outcomes,
                        final_k_outcomes=final_k_outcomes,
                        required_terms=required_terms,
                        mutation_boost_terms=mutation_boost_terms,
                        drug_keywords=drug_keywords
                    )
                else:
                    print(f"Unsupported source type: {source_type}")
                    continue
                    
                results[source_type] = extracted_picos
            
            # Run consolidation if requested
            if run_consolidation:
                print("\n=== Running PICO and Outcomes Consolidation ===")
                consolidation_results = self.run_pico_consolidation(
                    source_types=source_types
                )
                results["consolidation"] = consolidation_results
            
            return results
            
        finally:
            # Restore original case if it was temporarily changed
            if case is not None:
                self.case = original_case
                if self.case:
                    self.path_results = os.path.join("results", self.case)
                    self.path_chunks = os.path.join(self.path_results, "chunks")
                    self.path_pico = os.path.join(self.path_results, "PICO")
                    self.path_consolidated = os.path.join(self.path_results, "consolidated")
                else:
                    self.path_results = "results"
                    self.path_chunks = os.path.join(self.path_results, "chunks")
                    self.path_pico = os.path.join(self.path_results, "PICO")
                    self.path_consolidated = os.path.join(self.path_results, "consolidated")