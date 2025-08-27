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
    5. Retrieval (now supports split retrieval for Population & Comparator vs Outcomes)
    6. PICO extraction

    Enhanced to support different source types with specialized retrieval strategies
    and configurable filtering parameters including mutation-specific retrieval.
    Now supports separate retrieval pipelines for Population & Comparator vs Outcomes.
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
        source_type_configs: Optional[Dict[str, Any]] = None
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
            source_type_configs: Configuration for different source types
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.vectorstore_type = vectorstore_type

        os.makedirs(self.path_results, exist_ok=True)
        os.makedirs(self.path_chunks, exist_ok=True)
        os.makedirs(self.path_pico, exist_ok=True)

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

        self.source_type_configs = source_type_configs or {}

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

    def vectorize_documents(self, embeddings_type: Optional[str] = None):
        """
        Vectorize chunked documents using specified embedding type.
        
        Args:
            embeddings_type: Type of embeddings to use ("openai", "biobert", or "both")
                           If None, uses the value specified in self.vectorstore_type
        """
        if embeddings_type is None:
            embeddings_type = self.vectorstore_type
            
        os.makedirs(self.path_vectorstore, exist_ok=True)
        
        if embeddings_type.lower() in ["openai", "both"]:
            self.vectoriser_openai = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="openai",
                db_parent_dir=self.path_vectorstore
            )
            self.vectorstore_openai = self.vectoriser_openai.run_pipeline()
            print("Created OpenAI vectorstore")
            
        if embeddings_type.lower() in ["biobert", "both"]:
            self.vectoriser_biobert = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="biobert",
                db_parent_dir=self.path_vectorstore
            )
            self.vectorstore_biobert = self.vectoriser_biobert.run_pipeline()
            print("Created BioBERT vectorstore")
            
        if embeddings_type.lower() == "both" and self.vectoriser_openai and self.vectoriser_biobert:
            print("Visualizing vectorstore comparison")
            self.vectoriser_openai.visualize_vectorstore(self.vectorstore_biobert)

    def initialize_retriever(self, vectorstore_type: Optional[str] = None):
        """
        Initialize the retriever with the specified vectorstore.
        
        Args:
            vectorstore_type: Type of vectorstore to use ("openai" or "biobert")
                            If None, uses the value specified in self.vectorstore_type
        """
        if vectorstore_type is None:
            vectorstore_type = self.vectorstore_type
            
        if vectorstore_type.lower() == "openai" and self.vectorstore_openai:
            self.retriever = ChunkRetriever(vectorstore=self.vectorstore_openai, results_output_dir=self.path_results)
        elif vectorstore_type.lower() == "biobert" and self.vectorstore_biobert:
            self.retriever = ChunkRetriever(vectorstore=self.vectorstore_biobert, results_output_dir=self.path_results)
        else:
            if vectorstore_type.lower() == "openai":
                print("OpenAI vectorstore not available. Please run vectorize_documents first.")
            else:
                print("BioBERT vectorstore not available. Please run vectorize_documents first.")
            return
        
        print(f"Initialized retriever with {vectorstore_type} vectorstore and specialized retrieval methods")

    def initialize_pico_extractors(self):
        """Initialize separate PICO extractors for HTA and clinical guideline sources."""
        if not self.source_type_configs:
            print("Source type configurations not provided. Cannot initialize PICO extractors.")
            return
            
        if "hta_submission" in self.source_type_configs:
            hta_config = self.source_type_configs["hta_submission"]
            self.pico_extractor_hta = PICOExtractor(
                system_prompt=hta_config["system_prompt"],
                user_prompt_template=hta_config["user_prompt_template"],
                source_type="hta_submission",
                model_name=self.model,
                results_output_dir=self.path_results
            )
        
        if "clinical_guideline" in self.source_type_configs:
            clinical_config = self.source_type_configs["clinical_guideline"]
            self.pico_extractor_clinical = PICOExtractor(
                system_prompt=clinical_config["system_prompt"],
                user_prompt_template=clinical_config["user_prompt_template"],
                source_type="clinical_guideline",
                model_name=self.model,
                results_output_dir=self.path_results
            )
        
        print(f"Initialized specialized PICO extractors for available source types with model {self.model}")

    def get_all_countries(self):
        """
        Retrieves all unique country codes available in the vectorstore.
        
        Returns:
            List[str]: List of unique country codes
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return []
        
        result = self.retriever.chroma_collection.get(
            limit=10000,
            include=['metadatas']
        )
        
        countries = set()
        for metadata in result['metadatas']:
            if metadata and 'country' in metadata and metadata['country'] not in ['unknown', None, '']:
                countries.add(metadata['country'])
        
        return sorted(list(countries))

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
            print(f"Retrieving {source_type} population/comparator chunks for all available countries: {', '.join(countries)}")
        
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

        return {
            "summary": {country: len(chunks) for country, chunks in chunks_by_country.items()},
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
            print(f"Retrieving {source_type} outcomes chunks for all available countries: {', '.join(countries)}")
        
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

        return {
            "summary": {country: len(chunks) for country, chunks in chunks_by_country.items()},
            "chunks_by_country": chunks_by_country,
            "timestamp": timestamp
        }

    def run_pico_extraction_for_source_type(
        self,
        source_type: str,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ):
        """
        Run PICO extraction for a specific source type using pre-stored chunks.
        """
        if self.pico_extractor_hta is None or self.pico_extractor_clinical is None:
            self.initialize_pico_extractors()
        
        if source_type == "hta_submission":
            extractor = self.pico_extractor_hta
        elif source_type == "clinical_guideline":
            extractor = self.pico_extractor_clinical
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")
            
        extracted_picos = extractor.extract_picos(
            indication=indication,
            model_override=model_override
        )
        
        return extracted_picos

    def run_split_retrieval_for_source_type(
        self,
        source_type: str,
        countries: List[str],
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        indication: Optional[str] = None
    ):
        """
        Run both Population & Comparator and Outcomes retrieval for a source type.
        """
        results = {}
        
        print(f"Running Population & Comparator retrieval for {source_type}")
        pc_results = self.run_population_comparator_retrieval_for_source_type(
            source_type=source_type,
            countries=countries,
            initial_k=initial_k_pc,
            final_k=final_k_pc,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            indication=indication
        )
        results["population_comparator"] = pc_results
        
        print(f"Running Outcomes retrieval for {source_type}")
        outcomes_results = self.run_outcomes_retrieval_for_source_type(
            source_type=source_type,
            countries=countries,
            initial_k=initial_k_outcomes,
            final_k=final_k_outcomes,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            indication=indication
        )
        results["outcomes"] = outcomes_results
        
        return results

    def extract_picos_with_split_retrieval(
        self,
        countries: List[str],
        source_type: str = "hta_submission",
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        indication: Optional[str] = None
    ):
        """
        Extract PICOs using split retrieval approach (separate Population & Comparator vs Outcomes).
        """
        print(f"Step 1: Running split retrieval for {source_type}")
        retrieval_results = self.run_split_retrieval_for_source_type(
            source_type=source_type,
            countries=countries,
            initial_k_pc=initial_k_pc,
            final_k_pc=final_k_pc,
            initial_k_outcomes=initial_k_outcomes,
            final_k_outcomes=final_k_outcomes,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            indication=indication
        )
        
        if not retrieval_results:
            print("Split retrieval failed, cannot proceed with PICO extraction")
            return []
        
        print(f"Step 2: Running PICO extraction for {source_type}")
        extracted_picos = self.run_pico_extraction_for_source_type(
            source_type=source_type,
            indication=indication
        )
        
        return extracted_picos

    def extract_picos_hta_with_split_retrieval(
        self,
        countries: List[str],
        indication: str,
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        mutation_boost_terms: Optional[List[str]] = None
    ):
        """Extract PICOs from HTA submissions using split retrieval approach."""
        return self.extract_picos_with_split_retrieval(
            countries=countries,
            source_type="hta_submission",
            initial_k_pc=initial_k_pc,
            final_k_pc=final_k_pc,
            initial_k_outcomes=initial_k_outcomes,
            final_k_outcomes=final_k_outcomes,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            mutation_boost_terms=mutation_boost_terms,
            indication=indication
        )

    def extract_picos_clinical_with_split_retrieval(
        self,
        countries: List[str],
        indication: str,
        initial_k_pc: int = 70,
        final_k_pc: int = 18,
        initial_k_outcomes: int = 60,
        final_k_outcomes: int = 12,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None
    ):
        """Extract PICOs from clinical guidelines using split retrieval approach."""
        return self.extract_picos_with_split_retrieval(
            countries=countries,
            source_type="clinical_guideline",
            initial_k_pc=initial_k_pc,
            final_k_pc=final_k_pc,
            initial_k_outcomes=initial_k_outcomes,
            final_k_outcomes=final_k_outcomes,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            indication=indication
        )

    def diagnose_vectorstore(self, limit: int = 100):
        """
        Diagnose the vectorstore contents and metadata structure.
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return None
        
        return self.retriever.diagnose_vectorstore(limit=limit)
    
    def test_simple_retrieval(self, country: str = "EN", limit: int = 5):
        """
        Test simple retrieval functionality.
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return False
        
        return self.retriever.test_simple_retrieval(country=country, limit=limit)

    def run_case_based_pipeline_with_split_retrieval(
        self,
        case_config: Dict[str, Any],
        countries: List[str],
        source_types: List[str] = ["hta_submission", "clinical_guideline"],
        initial_k_pc: int = 50,
        final_k_pc: int = 20,
        initial_k_outcomes: int = 40,
        final_k_outcomes: int = 15,
        skip_processing: bool = True,
        skip_translation: bool = True,
        vectorstore_type: Optional[str] = None
    ):
        """
        Run pipeline using split retrieval for a specific case configuration.
        
        Args:
            case_config: Case configuration dictionary with 'indication' key
            countries: List of country codes or ["ALL"]
            source_types: List of source types to process
            initial_k_pc: Initial retrieval count for Population & Comparator
            final_k_pc: Final retrieval count for Population & Comparator
            initial_k_outcomes: Initial retrieval count for Outcomes
            final_k_outcomes: Final retrieval count for Outcomes
            skip_processing: Skip PDF processing
            skip_translation: Skip translation
            vectorstore_type: Vectorstore type to use
        
        Returns:
            Dictionary with extracted PICOs for each source type
        """
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
                extracted_picos = self.extract_picos_hta_with_split_retrieval(
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
                extracted_picos = self.extract_picos_clinical_with_split_retrieval(
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
        
        return results