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
        self.path_outcomes = os.path.join(results_path, "outcomes")
        self.path_consolidated = os.path.join(results_path, "consolidated")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.vectorstore_type = vectorstore_type
        self.case = case

        if self.case:
            self.path_results = os.path.join(results_path, self.case)
            self.path_chunks = os.path.join(self.path_results, "chunks")
            self.path_pico = os.path.join(self.path_results, "PICO")
            self.path_outcomes = os.path.join(self.path_results, "outcomes")
            self.path_consolidated = os.path.join(self.path_results, "consolidated")

        os.makedirs(self.path_results, exist_ok=True)
        os.makedirs(self.path_chunks, exist_ok=True)
        os.makedirs(self.path_pico, exist_ok=True)
        os.makedirs(self.path_outcomes, exist_ok=True)
        os.makedirs(self.path_consolidated, exist_ok=True)

        self.openai = OpenAI()

        self.translator = None
        self.chunker = None
        
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

    def initialize_pico_extractors(self, temperature: Optional[float] = None):
            """
            Initialize separate PICO extractors for HTA and clinical guideline sources.
            
            Args:
                temperature: LLM temperature for extraction. If None, uses 0.3 as default.
            """
            if not self.source_type_configs:
                print("Source type configurations not provided. Cannot initialize PICO extractors.")
                return
            
            if temperature is None:
                temperature = 0.3
                
            case_info = f" for case '{self.case}'" if self.case else ""
            
            if "hta_submission" in self.source_type_configs:
                hta_config = self.source_type_configs["hta_submission"]
                self.pico_extractor_hta = PICOExtractor(
                    system_prompt=hta_config.get("population_comparator_system_prompt", ""),
                    user_prompt_template=hta_config.get("population_comparator_user_prompt_template", ""),
                    source_type="hta_submission",
                    model_name=self.model,
                    results_output_dir=self.path_results,
                    source_type_config=hta_config,
                    temperature=temperature
                )
            
            if "clinical_guideline" in self.source_type_configs:
                clinical_config = self.source_type_configs["clinical_guideline"]
                self.pico_extractor_clinical = PICOExtractor(
                    system_prompt=clinical_config.get("population_comparator_system_prompt", ""),
                    user_prompt_template=clinical_config.get("population_comparator_user_prompt_template", ""),
                    source_type="clinical_guideline",
                    model_name=self.model,
                    results_output_dir=self.path_results,
                    source_type_config=clinical_config,
                    temperature=temperature
                )
            
            print(f"Initialized specialized PICO extractors for available source types with model {self.model} and temperature {temperature}{case_info}")

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
        Run PICO extraction (Population & Comparator only) for a specific source type.
        
        Args:
            source_type: Source type to extract PICOs for
            indication: Medical indication
            countries: List of countries (optional)
            
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
        
        import glob
        pattern = f"{source_type}_population_comparator_*_retrieval_results.json"
        pc_files = glob.glob(os.path.join(self.path_chunks, pattern))
        
        if not pc_files:
            print(f"No population/comparator retrieval result files found for {source_type}{case_info}. Please run retrieval first.")
            return None
            
        population_comparator_file = max(pc_files, key=os.path.getmtime)
        print(f"Extracting PICOs from {source_type}{case_info}...")
        print(f"Using population/comparator data from: {os.path.basename(population_comparator_file)}")
        
        results = extractor.extract_population_comparator(
            source_type=source_type,
            indication=indication
        )
        
        return results

    def run_outcomes_extraction_for_source_type(
        self,
        source_type: str,
        indication: str,
        countries: Optional[List[str]] = None
    ):
        """
        Run Outcomes extraction for a specific source type.
        
        Args:
            source_type: Source type to extract outcomes for
            indication: Medical indication
            countries: List of countries (optional)
            
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
        
        import glob
        pattern = f"{source_type}_outcomes_*_retrieval_results.json"
        outcomes_files = glob.glob(os.path.join(self.path_chunks, pattern))
        
        if not outcomes_files:
            print(f"No outcomes retrieval result files found for {source_type}{case_info}. Please run retrieval first.")
            return None
            
        outcomes_file = max(outcomes_files, key=os.path.getmtime)
        print(f"Extracting Outcomes from {source_type}{case_info}...")
        print(f"Using outcomes data from: {os.path.basename(outcomes_file)}")
        
        results = extractor.extract_outcomes(
            source_type=source_type,
            indication=indication
        )
        
        return results

    def run_pico_consolidation(
        self,
        source_types: List[str],
        indication: Optional[str] = None,
        test_set: bool = False
    ):
        """
        Run PICO and outcomes consolidation across multiple source types.
        
        Args:
            source_types: List of source types to consolidate
            indication: Medical indication (optional)
            test_set: If True, use test countries; if False, use train countries
            
        Returns:
            Dictionary with consolidation results
        """
        if self.pico_consolidator is None:
            print("PICO consolidator not initialized. Please run initialize_pico_consolidator first.")
            return None
        
        split_name = "test" if test_set else "train"
        case_info = f" for case '{self.case}'" if self.case else ""
        print(f"Running PICO and outcomes consolidation across {len(source_types)} source types{case_info} ({split_name} set)...")
        
        existing_source_types = []
        for source_type in source_types:
            pico_file = os.path.join(self.path_pico, f"{source_type}_picos.json")
            outcomes_file = os.path.join(self.path_outcomes, f"{source_type}_outcomes.json")
            
            has_pico = os.path.exists(pico_file)
            has_outcomes = os.path.exists(outcomes_file)
            
            if has_pico:
                print(f"Found PICO file for {source_type}: {os.path.basename(pico_file)}")
            if has_outcomes:
                print(f"Found Outcomes file for {source_type}: {os.path.basename(outcomes_file)}")
            
            if has_pico or has_outcomes:
                existing_source_types.append(source_type)
            else:
                print(f"Warning: No extraction files found for {source_type}")
        
        if not existing_source_types:
            print(f"No extraction files found for consolidation{case_info}. Please run extraction first.")
            return None
        
        results = self.pico_consolidator.consolidate_all(
            source_types=existing_source_types,
            test_set=test_set
        )
        
        return results

    def extract_picos_hta_with_retrieval(
        self,
        countries: List[str],
        indication: str,
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        drug_keywords: Optional[List[str]] = None,
        mutation_boost_terms: Optional[List[str]] = None
    ):
        """
        Extract PICOs from HTA submissions using retrieval pipeline.
        
        Args:
            countries: List of country codes
            indication: Medical indication
            initial_k_pc: Initial retrieval count for Population & Comparator
            final_k_pc: Final retrieval count for Population & Comparator
            initial_k_outcomes: Initial retrieval count for Outcomes
            final_k_outcomes: Final retrieval count for Outcomes
            drug_keywords: List of drug keywords for filtering
            mutation_boost_terms: List of mutation terms for boosting
            
        Returns:
            Dictionary with extraction results
        """
        self.run_population_comparator_retrieval_for_source_type(
            source_type="hta_submission",
            countries=countries,
            indication=indication,
            initial_k=initial_k_pc,
            final_k=final_k_pc,
            drug_keywords=drug_keywords,
            mutation_boost_terms=mutation_boost_terms
        )
        
        self.run_outcomes_retrieval_for_source_type(
            source_type="hta_submission",
            countries=countries,
            indication=indication,
            initial_k=initial_k_outcomes,
            final_k=final_k_outcomes,
            drug_keywords=drug_keywords,
            mutation_boost_terms=mutation_boost_terms
        )
        
        pico_results = self.run_pico_extraction_for_source_type(
            source_type="hta_submission",
            indication=indication,
            countries=countries
        )
        
        outcomes_results = self.run_outcomes_extraction_for_source_type(
            source_type="hta_submission",
            indication=indication,
            countries=countries
        )
        
        return {
            "picos": pico_results,
            "outcomes": outcomes_results
        }

    def extract_picos_clinical_with_retrieval(
        self,
        countries: List[str],
        indication: str,
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None
    ):
        """
        Extract PICOs from clinical guidelines using retrieval pipeline.
        
        Args:
            countries: List of country codes
            indication: Medical indication
            initial_k_pc: Initial retrieval count for Population & Comparator
            final_k_pc: Final retrieval count for Population & Comparator
            initial_k_outcomes: Initial retrieval count for Outcomes
            final_k_outcomes: Final retrieval count for Outcomes
            required_terms: List of required term combinations
            mutation_boost_terms: List of mutation terms for boosting
            drug_keywords: List of drug keywords for filtering
            
        Returns:
            Dictionary with extraction results
        """
        self.run_population_comparator_retrieval_for_source_type(
            source_type="clinical_guideline",
            countries=countries,
            indication=indication,
            initial_k=initial_k_pc,
            final_k=final_k_pc,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            drug_keywords=drug_keywords
        )
        
        self.run_outcomes_retrieval_for_source_type(
            source_type="clinical_guideline",
            countries=countries,
            indication=indication,
            initial_k=initial_k_outcomes,
            final_k=final_k_outcomes,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            drug_keywords=drug_keywords
        )
        
        pico_results = self.run_pico_extraction_for_source_type(
            source_type="clinical_guideline",
            indication=indication,
            countries=countries
        )
        
        outcomes_results = self.run_outcomes_extraction_for_source_type(
            source_type="clinical_guideline",
            indication=indication,
            countries=countries
        )
        
        return {
            "picos": pico_results,
            "outcomes": outcomes_results
        }

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
        run_consolidation: bool = False,
        test_set: bool = False
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
            test_set: If True, use test countries; if False, use train countries
        
        Returns:
            Dictionary with extracted PICOs for each source type and consolidation results
        """
        if case is not None:
            original_case = self.case
            self.case = case
            self.path_results = os.path.join("results", self.case)
            self.path_chunks = os.path.join(self.path_results, "chunks")
            self.path_pico = os.path.join(self.path_results, "PICO")
            self.path_outcomes = os.path.join(self.path_results, "outcomes")
            self.path_consolidated = os.path.join(self.path_results, "consolidated")
            
            os.makedirs(self.path_results, exist_ok=True)
            os.makedirs(self.path_chunks, exist_ok=True)
            os.makedirs(self.path_pico, exist_ok=True)
            os.makedirs(self.path_outcomes, exist_ok=True)
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
                    extracted_data = self.extract_picos_hta_with_retrieval(
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
                    extracted_data = self.extract_picos_clinical_with_retrieval(
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
                    
                results[source_type] = extracted_data
            
            if run_consolidation:
                split_name = "test" if test_set else "train"
                print(f"\n=== Running PICO and Outcomes Consolidation for {split_name} set ===")
                consolidation_results = self.run_pico_consolidation(
                    source_types=source_types,
                    test_set=test_set
                )
                results["consolidation"] = consolidation_results
            
            return results
            
        finally:
            if case is not None:
                self.case = original_case
                if self.case:
                    self.path_results = os.path.join("results", self.case)
                    self.path_chunks = os.path.join(self.path_results, "chunks")
                    self.path_pico = os.path.join(self.path_results, "PICO")
                    self.path_outcomes = os.path.join(self.path_results, "outcomes")
                    self.path_consolidated = os.path.join(self.path_results, "consolidated")
                else:
                    self.path_results = "results"
                    self.path_chunks = os.path.join(self.path_results, "chunks")
                    self.path_pico = os.path.join(self.path_results, "PICO")
                    self.path_outcomes = os.path.join(self.path_results, "outcomes")
                    self.path_consolidated = os.path.join(self.path_results, "consolidated")


class SimulationRunner:
    """
    Manages simulation execution by determining which paths need to be created
    based on what differs from the baseline configuration.
    """
    
    def __init__(
        self,
        simulation_id: str,
        base_paths: Optional[Dict[str, str]] = None,
        case: Optional[str] = None
    ):
        """
        Initialize simulation runner.
        
        Args:
            simulation_id: Simulation identifier (e.g., "base", "sim_1", "sim_2")
            base_paths: Base directory paths for data storage
            case: Medical case identifier (e.g., "nsclc", "hcc")
        """
        from python.config import SIMULATION_CONFIGS
        
        if simulation_id not in SIMULATION_CONFIGS:
            raise ValueError(f"Unknown simulation_id: {simulation_id}. Available: {list(SIMULATION_CONFIGS.keys())}")
        
        self.simulation_id = simulation_id
        self.case = case
        self.config = SIMULATION_CONFIGS[simulation_id]
        self.base_config = SIMULATION_CONFIGS["base"]
        
        if base_paths is None:
            base_paths = {
                "chunked": "data/text_chunked",
                "vectorstore": "data/vectorstore",
                "results": "results"
            }
        
        self.base_paths = base_paths
        self.paths = self._determine_paths()
        
    def _needs_new_vectorstore(self) -> bool:
        """
        Determines if this simulation needs a new vectorstore
        (different chunk parameters from baseline).
        """
        return (
            self.config["chunk_params"]["min_chunk_size"] != self.base_config["chunk_params"]["min_chunk_size"] or
            self.config["chunk_params"]["max_chunk_size"] != self.base_config["chunk_params"]["max_chunk_size"]
        )
    
    def _needs_new_chunks(self) -> bool:
        """
        Determines if this simulation needs new chunk retrieval
        (different retrieval params or chunk params from baseline).
        """
        if self._needs_new_vectorstore():
            return True
            
        base_retrieval = self.base_config["retrieval_params"]
        sim_retrieval = self.config["retrieval_params"]
        
        for source_type in ["hta_submission", "clinical_guideline"]:
            for retrieval_type in ["population_comparator", "outcomes"]:
                if base_retrieval[source_type][retrieval_type] != sim_retrieval[source_type][retrieval_type]:
                    return True
        
        return False
    
    def _needs_new_extraction(self) -> bool:
        """
        Determines if this simulation needs new PICO extraction
        (different temperature, retrieval params, or chunk params from baseline).
        """
        if self._needs_new_chunks():
            return True
            
        return self.config["extraction_temperature"] != self.base_config["extraction_temperature"]
    

    def _determine_paths(self) -> Dict[str, str]:
        """
        Determines which paths to use for this simulation.
        Creates simulation-specific folders where needed.
        Each simulation ALWAYS gets its own results folder to keep outputs separate.
        """
        paths = {}
        
        # Vectorstore and chunks: only create new if chunk params differ from base
        if self._needs_new_vectorstore():
            if self.case:
                paths["chunked"] = os.path.join(self.base_paths["chunked"], self.simulation_id, self.case)
                paths["vectorstore"] = os.path.join(self.base_paths["vectorstore"], self.simulation_id)
            else:
                paths["chunked"] = os.path.join(self.base_paths["chunked"], self.simulation_id)
                paths["vectorstore"] = os.path.join(self.base_paths["vectorstore"], self.simulation_id)
        else:
            # Reuse base vectorstore and chunks if params are the same
            if self.case:
                paths["chunked"] = os.path.join(self.base_paths["chunked"], "base", self.case)
                paths["vectorstore"] = os.path.join(self.base_paths["vectorstore"], "base")
            else:
                paths["chunked"] = os.path.join(self.base_paths["chunked"], "base")
                paths["vectorstore"] = os.path.join(self.base_paths["vectorstore"], "base")
        
        # Results: ALWAYS create a separate folder for each simulation
        # This ensures each simulation run keeps its own outputs separate
        if self.case:
            paths["results"] = os.path.join(self.base_paths["results"], self.simulation_id, self.case)
        else:
            paths["results"] = os.path.join(self.base_paths["results"], self.simulation_id)
        
        return paths
        
    def create_folders(self):
        """
        Creates necessary top-level folder structure for this simulation.
        Case-specific folders (PICO, chunks, outcomes, consolidated) are created
        automatically by RagPipeline when case is specified.
        """
        os.makedirs(self.paths["chunked"], exist_ok=True)
        os.makedirs(self.paths["vectorstore"], exist_ok=True)
        os.makedirs(self.paths["results"], exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"SIMULATION: {self.simulation_id}")
        print(f"{'='*80}")
        print(f"Name: {self.config['name']}")
        print(f"Description: {self.config['description']}")
        print(f"\nConfiguration:")
        print(f"  Extraction Temperature: {self.config['extraction_temperature']}")
        print(f"  Chunk Size: min={self.config['chunk_params']['min_chunk_size']}, max={self.config['chunk_params']['max_chunk_size']}")
        print(f"\nRetrieval Parameters:")
        for source_type, params in self.config['retrieval_params'].items():
            print(f"  {source_type}:")
            for retrieval_type, values in params.items():
                print(f"    {retrieval_type}: initial_k={values['initial_k']}, final_k={values['final_k']}")
        print(f"\nPath Configuration:")
        print(f"  Needs new vectorstore: {self._needs_new_vectorstore()}")
        print(f"  Needs new chunks: {self._needs_new_chunks()}")
        print(f"  Needs new extraction: {self._needs_new_extraction()}")
        print(f"  Chunked path: {self.paths['chunked']}")
        print(f"  Vectorstore path: {self.paths['vectorstore']}")
        print(f"  Results path: {self.paths['results']}")
        print(f"{'='*80}\n")
    
    def get_paths(self) -> Dict[str, str]:
        """
        Returns the path configuration for this simulation.
        """
        return self.paths
    
    def get_chunk_params(self) -> Dict[str, int]:
        """
        Returns chunk parameters for this simulation.
        """
        return self.config["chunk_params"]
    
    def get_retrieval_params(self, source_type: str, retrieval_type: str) -> Dict[str, int]:
        """
        Returns retrieval parameters for a specific source and retrieval type.
        
        Args:
            source_type: "hta_submission" or "clinical_guideline"
            retrieval_type: "population_comparator" or "outcomes"
        """
        return self.config["retrieval_params"][source_type][retrieval_type]
    
    def get_extraction_temperature(self) -> float:
        """
        Returns extraction temperature for this simulation.
        """
        return self.config["extraction_temperature"]
    
    def print_summary(self):
        """
        Prints a summary of what this simulation will test.
        """
        print(f"\n{'='*80}")
        print(f"SIMULATION SUMMARY: {self.simulation_id}")
        print(f"{'='*80}")
        print(f"{self.config['name']}")
        print(f"\n{self.config['description']}")
        
        if self._needs_new_vectorstore():
            print(f"\nVectorstore: NEW (different chunk sizes)")
        else:
            print(f"\nVectorstore: REUSING baseline")
        
        if self._needs_new_chunks():
            if self._needs_new_vectorstore():
                print(f"Retrieval: NEW (different vectorstore)")
            else:
                print(f"Retrieval: NEW (different retrieval parameters)")
        else:
            print(f"Retrieval: REUSING baseline")
        
        if self._needs_new_extraction():
            if self._needs_new_chunks():
                print(f"Extraction: NEW (different retrieval)")
            else:
                print(f"Extraction: NEW (different temperature)")
        else:
            print(f"Extraction: REUSING baseline")
        
        print(f"{'='*80}\n")