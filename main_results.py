from python.utils import FolderTree
from python.process import PDFProcessor, PostCleaner
from python.translation import Translator
from python.vectorise import Chunker, Vectoriser
from python.run import RagPipeline
from python.open_ai import validate_api_key
from python.config import SOURCE_TYPE_CONFIGS, CASE_CONFIGS, CONSOLIDATION_CONFIGS
from python.results_descriptive import RunResults
import glob
import os
from pathlib import Path

PDF_PATH = "data/PDF"
CLEAN_PATH = "data/text_cleaned"
TRANSLATED_PATH = "data/text_translated"
POST_CLEANED_PATH = "data/post_cleaned"
CHUNKED_PATH = "data/text_chunked"
VECTORSTORE_PATH = "data/vectorstore"
VECTORSTORE_TYPE = "biobert"
MODEL = "gpt-4.1"
COUNTRIES = ["ALL"]

CASES = ["nsclc", "hcc"]

# Define which simulations to run and analyze
SIMULATION_IDS = ["base", "base_b", "base_c", "base_d", "base_e"]
