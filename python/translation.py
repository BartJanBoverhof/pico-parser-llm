import os
import re
import json
import shutil
import gc
import torch
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import copy

class Translator:
    """
    Enhanced Translator class with simplified medical term preservation,
    improved chunking strategy, and better quality assessment.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0
        self.total_translation_start_time = None  

        self.processing_start_time = None

        # Simplified medical term whitelist - only truly critical terms
        self.critical_medical_terms = {
            # Drug names that should never be translated
            'drug_names': [
                'sotorasib', 'lumykras', 'pembrolizumab', 'keytruda', 'atezolizumab',
                'nivolumab', 'durvalumab', 'ipilimumab', 'bevacizumab', 'cetuximab',
                'panitumumab', 'trastuzumab', 'pertuzumab', 'ramucirumab',
                'nintedanib', 'osimertinib', 'erlotinib', 'gefitinib', 'afatinib',
                'crizotinib', 'alectinib', 'brigatinib', 'lorlatinib',
                'pemetrexed', 'docetaxel', 'paclitaxel', 'carboplatin', 'cisplatin',
                'gemcitabine', 'vinorelbine', 'etoposide'
            ],
            # Medical abbreviations
            'abbreviations': [
                'NSCLC', 'NDRP', 'SCLC', 'EGFR', 'ALK', 'ROS1', 'KRAS', 'BRAF',
                'PD-L1', 'TMB', 'MSI', 'MMR', 'HER2', 'MET', 'RET', 'NTRK',
                'PFS', 'OS', 'ORR', 'DCR', 'DOR', 'TTR', 'TTP', 'QoL',
                'ECOG', 'KPS', 'PS', 'CR', 'PR', 'SD', 'PD',
                'AE', 'SAE', 'TEAE', 'CTCAE', 'NCI', 'WHO', 'RECIST',
                'ITT', 'PP', 'mITT', 'FAS', 'SS'
            ],
            # Regulatory/organizational terms
            'regulatory': [
                'FDA', 'EMA', 'NICE', 'SMC', 'G-BA', 'HAS', 'AIFA', 'TGA', 'PMDA',
                'CADTH', 'IQWIG', 'HTA', 'QALY', 'ICER', 'ICUR'
            ]
        }

        # Translation models - facebook/nllb-200-3.3B as primary with fallbacks
        self.translation_models = [
            'facebook/nllb-200-3.3B',
            'facebook/nllb-200-1.3B',
            'facebook/nllb-200-distilled-1.3B',
            'facebook/nllb-200-distilled-600M'
        ]

        # Language mapping for NLLB models
        self.nllb_lang_mapping = {
            'en': 'eng_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn',
            'it': 'ita_Latn', 'pt': 'por_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn',
            'ru': 'rus_Cyrl', 'zh': 'zho_Hans', 'ja': 'jpn_Jpan', 'ar': 'ara_Arab',
            'hi': 'hin_Deva', 'bg': 'bul_Cyrl', 'cs': 'ces_Latn', 'da': 'dan_Latn',
            'fi': 'fin_Latn', 'el': 'ell_Grek', 'hu': 'hun_Latn', 'ro': 'ron_Latn',
            'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sv': 'swe_Latn', 'uk': 'ukr_Cyrl',
            'hr': 'hrv_Latn', 'no': 'nno_Latn', 'et': 'est_Latn', 'lv': 'lav_Latn',
            'lt': 'lit_Latn', 'tr': 'tur_Latn', 'he': 'heb_Hebr', 'th': 'tha_Thai',
            'ko': 'kor_Hang', 'vi': 'vie_Latn', 'fa': 'fas_Arab', 'sr': 'srp_Cyrl',
            'ca': 'cat_Latn', 'mt': 'mlt_Latn', 'cy': 'cym_Latn', 'is': 'isl_Latn',
        }

        # Quality thresholds
        self.quality_thresholds = {
            'minimum_acceptable_quality': 0.40,
            'medical_preservation_threshold': 0.70,
        }

        # Improved chunking parameters for medical documents
        self.chunk_params = {
            'max_chars': 2500,
            'overlap_chars': 150,
            'min_chunk_chars': 300,
        }

        # Table detection patterns
        self.table_patterns = [
            r'(?:Row|Column)\s+\d+[:\s]',
            r'Table\s+(?:Title|contains|shows)',
            r'\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|',
            r'€\s*\d+(?:[.,]\d+)*',
            r'\d+(?:[.,]\d+)*\s*€',
        ]

        # CUDA setup
        self.use_cuda = False
        self.device = "cpu"

        if torch.cuda.is_available():
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    gc.collect()

                test_tensor = torch.tensor([1.0], dtype=torch.float32)
                test_tensor = test_tensor.to('cuda:0')
                result = test_tensor + 1
                test_tensor = test_tensor.cpu()
                del test_tensor, result
                torch.cuda.empty_cache()

                self.use_cuda = True
                self.device = "cuda:0"
                print("✓ Using CUDA")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed, using CPU: {str(e)[:100]}")
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                gc.collect()
                self.device = "cpu"
        else:
            print("⚠️  CUDA not available, using CPU")

        # Current loaded translator info
        self.current_translator = None
        self.current_language = None
        self.current_model_name = None

    def extract_medical_terms_database(self, text: str) -> Dict[str, str]:
        """Simplified medical term extraction using whitelist approach."""
        if not text:
            return {}
            
        term_db = {}
        term_counter = 0
        
        # Combine all critical terms
        all_terms = []
        for category, terms in self.critical_medical_terms.items():
            all_terms.extend(terms)
        
        # Sort by length (longest first) to avoid partial matches
        all_terms.sort(key=len, reverse=True)
        
        text_lower = text.lower()
        
        for term in all_terms:
            term_lower = term.lower()
            # Simple case-insensitive search
            if term_lower in text_lower:
                # Find actual case-sensitive matches
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match not in term_db.values():
                        term_id = f"MEDTERM{term_counter:03d}"
                        term_db[term_id] = match
                        term_counter += 1
        
        return term_db

    def apply_medical_term_protection(self, text: str, term_db: Dict[str, str]) -> str:
        """Apply simplified term protection."""
        if not text or not term_db:
            return text
            
        protected_text = text
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(term_db.items(), key=lambda x: len(x[1]), reverse=True)
        
        for term_id, term in sorted_terms:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(term) + r'\b'
            protected_text = re.sub(pattern, term_id, protected_text, flags=re.IGNORECASE)
        
        return protected_text

    def restore_medical_terms(self, text: str, term_db: Dict[str, str]) -> str:
        """Restore terms with validation."""
        if not text or not term_db:
            return text
            
        restored_text = text
        unreplaced_count = 0
        
        for term_id, term in term_db.items():
            if term_id in restored_text:
                restored_text = restored_text.replace(term_id, term)
            else:
                unreplaced_count += 1
        
        # Log if restoration failed
        if unreplaced_count > 0:
            print(f"      ⚠️  {unreplaced_count} medical terms failed to restore")
        
        return restored_text

    def count_tokens_accurately(self, text: str) -> int:
        """Simplified token counting."""
        if not text:
            return 0
        
        # Simple approximation: 4 characters per token on average
        return len(text) // 4

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Improved text splitting with better sentence boundary detection."""
        if not text:
            return []
            
        max_chars = self.chunk_params['max_chars']
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while remaining_text:
            if len(remaining_text) <= max_chars:
                chunks.append(remaining_text)
                break
            
            # Find the best split point - prioritize sentence boundaries
            split_point = max_chars
            
            # Look for sentence endings first
            for i in range(max_chars, max_chars // 2, -1):
                if i >= len(remaining_text):
                    continue
                    
                char = remaining_text[i]
                # Strong sentence boundaries
                if char in '.!?' and i + 1 < len(remaining_text):
                    next_char = remaining_text[i + 1]
                    if next_char.isspace() or next_char.isupper():
                        split_point = i + 1
                        break
                # Paragraph breaks
                elif char == '\n' and remaining_text[i:i+2] == '\n\n':
                    split_point = i + 2
                    break
            
            # If no good sentence boundary found, look for word boundaries
            if split_point == max_chars:
                for i in range(max_chars, max_chars // 2, -1):
                    if i >= len(remaining_text):
                        continue
                    if remaining_text[i].isspace():
                        split_point = i
                        break
            
            chunk = remaining_text[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining_text = remaining_text[split_point:].strip()
        
        return [chunk for chunk in chunks if chunk.strip()]

    def adaptive_chunk_for_translation(self, text: str) -> List[str]:
        """Simplified chunking for medical documents."""
        if not text:
            return []
            
        # Check if chunking is needed
        if self.count_tokens_accurately(text) <= 500 and len(text) <= self.chunk_params['max_chars']:
            return [text]
        
        print(f"      🔨 Chunking needed for large text ({len(text)} chars)")
        
        chunks = self.split_text_into_chunks(text)
        
        print(f"      ✓ Created {len(chunks)} chunks with improved sentence boundaries")
        return chunks

    def is_table_content(self, text: str) -> bool:
        """Check if text is table content."""
        if not text:
            return False
            
        for pattern in self.table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for structural indicators
        line_count = len(text.split('\n'))
        pipe_density = text.count('|') / max(len(text), 1)
        numeric_density = len(re.findall(r'\d+(?:\.\d+)?', text)) / max(len(text.split()), 1)
        
        return (line_count >= 5 and pipe_density > 0.05) or numeric_density > 0.3

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        """Enhanced quality assessment with medical focus and English quality checks."""
        if not translated_text.strip():
            return {'overall': 0.0, 'english_quality': 0.0, 'medical_preservation': 0.0, 'has_placeholders': 1.0}
        
        # Check for unreplaced placeholders
        placeholder_score = 1.0
        if re.search(r'MEDTERM\d+', translated_text) or re.search(r'\[MED\d+\]', translated_text):
            placeholder_score = 0.0
            print(f"      ⚠️  Found unreplaced placeholders in translation")
        
        # Basic English quality checks
        english_quality = 1.0
        try:
            detected_lang = detect(translated_text)
            if detected_lang != 'en':
                english_quality *= 0.3
        except:
            english_quality *= 0.5
        
        # Check for basic English patterns
        if not re.search(r'\b(?:the|and|of|to|a|in|is|it|that|for|on|with|as)\b', translated_text.lower()):
            english_quality *= 0.5
        
        # Check for excessive repetition
        words = translated_text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                english_quality *= 0.5
        
        # Length ratio check
        len_ratio = len(translated_text) / max(len(original_text), 1)
        length_score = 1.0 if 0.5 <= len_ratio <= 2.5 else 0.5
        
        # Medical term preservation check (simplified)
        orig_terms = self.extract_medical_terms_database(original_text)
        medical_preservation = 1.0
        if orig_terms:
            preserved_count = 0
            for term in orig_terms.values():
                if term.lower() in translated_text.lower():
                    preserved_count += 1
            medical_preservation = preserved_count / len(orig_terms)
        
        # Overall weighted score
        overall_score = (
            placeholder_score * 0.30 +
            english_quality * 0.35 +
            medical_preservation * 0.25 +
            length_score * 0.10
        )
        
        return {
            'overall': overall_score,
            'english_quality': english_quality,
            'medical_preservation': medical_preservation,
            'has_placeholders': 1.0 - placeholder_score,
            'length_score': length_score
        }

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        """Document-level quality assessment."""
        if 'chunks' not in translated_data or 'chunks' not in original_data:
            return {'overall': 0.0, 'chunk_count': 0}
        
        chunk_qualities = []
        
        for orig_chunk, trans_chunk in zip(original_data['chunks'], translated_data['chunks']):
            if 'text' in orig_chunk and 'text' in trans_chunk:
                if orig_chunk['text'] and trans_chunk['text']:
                    text_quality = self.assess_translation_quality(
                        orig_chunk['text'], trans_chunk['text'], language
                    )
                    chunk_qualities.append(text_quality)
        
        if not chunk_qualities:
            return {'overall': 0.0, 'chunk_count': 0}
        
        # Calculate averages
        metrics = ['overall', 'english_quality', 'medical_preservation', 'has_placeholders', 'length_score']
        final_scores = {}
        
        for metric in metrics:
            scores = [quality[metric] for quality in chunk_qualities if metric in quality]
            final_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        final_scores['chunk_count'] = len(chunk_qualities)
        
        return final_scores

    def should_retranslate_with_higher_tier(self, quality_scores: Dict[str, float]) -> bool:
        """Check if retranslation is needed based on quality scores."""
        overall_quality = quality_scores.get('overall', 0.0)
        medical_score = quality_scores.get('medical_preservation', 0.0)
        has_placeholders = quality_scores.get('has_placeholders', 0.0)
        
        # Always retranslate if placeholders weren't restored
        if has_placeholders > 0.1:
            print(f"    📉 Unreplaced placeholders detected - retranslating")
            return True
        
        if overall_quality < self.quality_thresholds['minimum_acceptable_quality']:
            print(f"    📉 Overall quality {overall_quality:.3f} below threshold")
            return True
        
        if medical_score < self.quality_thresholds['medical_preservation_threshold']:
            print(f"    🏥 Medical preservation {medical_score:.3f} below threshold")
            return True
        
        return False

    def compare_translation_quality(self, quality_1: Dict[str, float], quality_2: Dict[str, float]) -> int:
        """Compare two quality scores with focus on medical preservation."""
        # Strongly penalize unreplaced placeholders
        if quality_1.get('has_placeholders', 0) < quality_2.get('has_placeholders', 0):
            return 1
        elif quality_1.get('has_placeholders', 0) > quality_2.get('has_placeholders', 0):
            return 2
        
        # Compare overall quality
        if abs(quality_1['overall'] - quality_2['overall']) > 0.05:
            return 1 if quality_1['overall'] > quality_2['overall'] else 2
        
        # Secondary: medical preservation
        med_diff = quality_1.get('medical_preservation', 0) - quality_2.get('medical_preservation', 0)
        if abs(med_diff) > 0.08:
            return 1 if med_diff > 0 else 2
        
        return 1

    def detect_document_language(self, text: str) -> Optional[str]:
        """Detect the primary language of a document."""
        if not text or len(text.strip()) < 20:
            return None

        try:
            clean_text = ' '.join(text.split()[:200])
            detected_lang = detect(clean_text)
            print(f"    Detected language: {detected_lang}")
            return detected_lang
        except Exception:
            print("    Language detection failed")
            return None

    def is_english_chunk(self, text: str) -> bool:
        """Quick check if a chunk is in English."""
        if not text or len(text.strip()) < 10:
            return True

        text_lower = text.lower()
        english_words = [
            ' the ', ' and ', ' of ', ' to ', ' a ', ' in ', ' is ', ' it ', ' you ', ' that ',
            ' he ', ' was ', ' for ', ' on ', ' are ', ' as ', ' with ', ' his ', ' they '
        ]

        english_count = sum(1 for word in english_words if word in f' {text_lower} ')
        total_words = len(text.split())

        if total_words > 5:
            english_ratio = english_count / total_words
            return english_ratio > 0.1

        return False

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"    ✗ Model {model_name} not available: {str(e)[:50]}")
            return False

    def get_available_models_for_language(self, language: str) -> List[str]:
        """Get available models for a language."""
        available_models = []
        
        if language in self.nllb_lang_mapping:
            for model_name in self.translation_models:
                if self.check_model_availability(model_name):
                    available_models.append(model_name)
        
        return available_models

    def load_nllb_model(self, model_name: str, language: str) -> Optional[Any]:
        """Load NLLB model."""
        try:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                gc.collect()

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )

            if self.device.startswith("cuda"):
                model = model.to(self.device)

            def nllb_translate(text, generation_params=None):
                try:
                    tgt_lang = 'eng_Latn'
                    max_input_length = 600
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    gen_kwargs = {
                        'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang),
                        'max_length': 400,
                        'num_beams': 4,
                        'length_penalty': 1.0,
                        'do_sample': False,
                        'no_repeat_ngram_size': 3,
                        'repetition_penalty': 1.2,
                    }

                    with torch.no_grad():
                        translated_tokens = model.generate(**inputs, **gen_kwargs)

                    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    return [{'translation_text': translation}]

                except Exception as e:
                    print(f"      NLLB translation error: {str(e)[:50]}")
                    return [{'translation_text': text}]

            print(f"    ✓ NLLB model loaded successfully on {self.device}")
            return nllb_translate

        except Exception as e:
            print(f"    ✗ NLLB model failed: {str(e)[:50]}")
            return None

    def load_translator_for_language(self, language: str):
        """Load translator for given language."""
        if self.current_language == language and self.current_translator:
            return self.current_translator

        self.clear_translator()

        print(f"  🔄 Loading translator for language: {language}")

        available_models = self.get_available_models_for_language(language)
        
        if not available_models:
            print(f"    ✗ No available models for language: {language}")
            return None
        
        for model_name in available_models:
            print(f"    Trying model: {model_name}")
            
            try:
                translator = self.load_nllb_model(model_name, language)
                
                if translator:
                    self.current_model_name = model_name
                    self.current_translator = translator
                    self.current_language = language
                    print(f"    ✓ Successfully loaded translator for {language}")
                    return translator
                    
            except Exception as e:
                print(f"    Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        print(f"    ✗ No translator could be loaded for language: {language}")
        return None

    def translate_single_chunk(self, text: str, translator) -> str:
        """Translate a single chunk using simplified medical term preservation."""
        if not text.strip() or not translator:
            return text

        try:
            # Extract medical terms and create simple database
            term_db = self.extract_medical_terms_database(text)
            
            # Apply protection if terms found
            protected_text = self.apply_medical_term_protection(text, term_db) if term_db else text
            
            # Check if chunking is needed
            chunks = self.adaptive_chunk_for_translation(protected_text)
            
            if len(chunks) == 1:
                # Single chunk translation
                gen_params = {
                    'max_length': 600,
                    'truncation': True,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 1.1,
                    'do_sample': False,
                    'num_beams': 4,
                }
                
                result = translator(protected_text, generation_params=gen_params)
                translated_text = result[0]['translation_text']
            else:
                # Multi-chunk translation
                translated_chunks = []
                for chunk in chunks:
                    gen_params = {
                        'max_length': 500,
                        'truncation': True,
                        'no_repeat_ngram_size': 3,
                        'repetition_penalty': 1.1,
                        'do_sample': False,
                        'num_beams': 4,
                    }
                    
                    result = translator(chunk, generation_params=gen_params)
                    translated_chunks.append(result[0]['translation_text'])
                
                translated_text = ' '.join(translated_chunks)
                print(f"      ✓ Merged {len(translated_chunks)} chunks")
            
            # Restore medical terms
            final_text = self.restore_medical_terms(translated_text, term_db) if term_db else translated_text
            
            return final_text.strip()

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def translate_document_with_tier(self, data: dict, language: str, tier: int) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        """Translate document."""
        print(f"  📝 Translating document")
        
        translation_start_time = datetime.now()

        translator = self.load_translator_for_language(language)
        if not translator:
            print(f"    ✗ No translator available")
            return data, {'overall': 0.0}, {
                'model_loaded': False,
                'processing_time_seconds': 0,
                'model_name': None
            }
        
        translated_data = copy.deepcopy(data)
        
        if 'chunks' not in translated_data:
            print(f"    No chunks found in document")
            return translated_data, {'overall': 0.0}, {
                'model_loaded': True,
                'processing_time_seconds': (datetime.now() - translation_start_time).total_seconds(),
                'model_name': self.current_model_name,
                'chunks_found': False
            }
        
        total_chunks = len(translated_data['chunks'])
        translated_count = 0
        english_count = 0
        table_count = 0
        
        print(f"    Processing {total_chunks} chunks with simplified medical term preservation...")
        
        for i, chunk in enumerate(translated_data['chunks']):
            if i % 20 == 0 or i == total_chunks - 1:
                print(f"      Chunk {i+1}/{total_chunks}")
            
            # Process heading
            if 'heading' in chunk and chunk['heading']:
                if self.is_english_chunk(chunk['heading']):
                    english_count += 1
                else:
                    chunk['heading'] = self.translate_single_chunk(
                        chunk['heading'], translator
                    )
                    translated_count += 1
            
            # Process text
            if 'text' in chunk and chunk['text']:
                if self.is_english_chunk(chunk['text']):
                    english_count += 1
                else:
                    if self.is_table_content(chunk['text']):
                        table_count += 1
                    
                    chunk['text'] = self.translate_single_chunk(
                        chunk['text'], translator
                    )
                    translated_count += 1
        
        processing_time = (datetime.now() - translation_start_time).total_seconds()
        
        print(f"    ✓ Translation complete:")
        print(f"      English chunks: {english_count}")
        print(f"      Translated chunks: {translated_count}")
        print(f"      Table chunks: {table_count}")
        
        # Quality assessment
        quality_scores = self.assess_document_quality(translated_data, data, language)
        
        print(f"    📊 Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      English Quality: {quality_scores.get('english_quality', 0):.3f}")
        print(f"      Medical Preservation: {quality_scores.get('medical_preservation', 0):.3f}")
        print(f"      Unreplaced Placeholders: {quality_scores.get('has_placeholders', 0):.3f}")
        
        translation_metadata = {
            'model_loaded': True,
            'model_name': self.current_model_name,
            'processing_time_seconds': processing_time,
            'chunks_found': True,
            'total_chunks': total_chunks,
            'chunks_translated': translated_count,
            'chunks_english': english_count,
            'table_chunks_processed': table_count,
            'quality_scores': quality_scores
        }
        
        return translated_data, quality_scores, translation_metadata

    def clear_translator(self):
        """Clear current translator and free memory."""
        self.current_translator = None
        self.current_language = None
        self.current_model_name = None

        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def process_json_file(self, input_path: str, output_path: str):
        """Process a single JSON file."""
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")
        
        self.processing_start_time = datetime.now()

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading JSON: {e}")
            return

        # Language detection
        all_text_parts = []
        if 'doc_id' in data:
            all_text_parts.append(str(data['doc_id']))

        if 'chunks' in data:
            for chunk in data['chunks']:
                if 'heading' in chunk and chunk['heading']:
                    all_text_parts.append(chunk['heading'])
                if 'text' in chunk and chunk['text']:
                    all_text_parts.append(chunk['text'])

        combined_text = ' '.join(all_text_parts)
        document_language = self.detect_document_language(combined_text)

        # Translation metadata
        translation_metadata = {
            "processing_timestamp": self.processing_start_time.isoformat(),
            "source_file": file_name,
            "detected_language": document_language,
            "was_translation_needed": False,
            "model_attempts": [],
            "final_model_used": None,
            "retranslation_occurred": False,
            "quality_comparison": {},
            "final_quality_scores": {},
            "chunks_translated": 0,
            "chunks_preserved_english": 0,
            "table_chunks_processed": 0,
            "total_processing_time_seconds": 0,
            "translation_strategy": "simplified_medical_optimized",
            "medical_preservation_used": "whitelist_approach",
            "table_content_detected": sum(1 for chunk in data.get('chunks', []) 
                                        if self.is_table_content(chunk.get('text', '')))
        }

        if not document_language or document_language == 'en':
            print(f"  📋 Document is English, copying without translation")
            translation_metadata.update({
                "translation_decision": "no_translation_needed_english",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Check model availability
        available_models = self.get_available_models_for_language(document_language)
        
        translation_metadata.update({
            "was_translation_needed": True,
            "models_available": len(available_models) > 0,
            "available_models": available_models
        })
        
        if not available_models:
            print(f"  📋 No translation models available for language {document_language}")
            translation_metadata.update({
                "translation_decision": "no_models_available",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Translate document
        print(f"  🎯 Starting translation with available models")
        final_translation, final_quality, translation_stats = self.translate_document_with_tier(
            data, document_language, 1
        )
        translation_metadata["model_attempts"].append(translation_stats)
        
        translation_metadata.update({
            "translation_decision": "translation_completed",
            "final_model_used": translation_stats["model_name"],
            "retranslation_occurred": False
        })

        # Final statistics
        translated_count = 0
        english_count = 0
        table_count = 0
        
        if 'chunks' in final_translation:
            for chunk in final_translation['chunks']:
                if 'text' in chunk and chunk['text']:
                    if self.is_table_content(chunk['text']):
                        table_count += 1
                
                for field in ['heading', 'text']:
                    if field in chunk and chunk[field]:
                        if self.is_english_chunk(chunk[field]):
                            english_count += 1
                        else:
                            translated_count += 1

        total_processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        translation_metadata.update({
            "final_quality_scores": final_quality,
            "chunks_translated": translated_count,
            "chunks_preserved_english": english_count,
            "table_chunks_processed": table_count,
            "total_processing_time_seconds": total_processing_time,
            "processing_completed_timestamp": datetime.now().isoformat()
        })

        final_translation["_translation_metadata"] = translation_metadata

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_translation, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Final result: {english_count} English, {translated_count} translated, {table_count} table chunks")
        print(f"  📊 Final quality: {final_quality['overall']:.3f}")
        print(f"  ⏱️  Processing time: {total_processing_time:.2f}s")

        self.english_chunks_preserved += english_count
        self.chunks_translated += translated_count
        self.clear_translator()

    def process_batch_files(self, file_batch: List[Tuple[str, str]]) -> None:
        """Process a batch of files."""
        print(f"  📦 Processing batch of {len(file_batch)} files")
        
        for input_path, output_path in file_batch:
            try:
                self.process_json_file(input_path, output_path)
            except Exception as e:
                print(f"  ✗ Batch processing error for {os.path.basename(input_path)}: {str(e)[:100]}")
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(input_path, output_path)
                    print(f"  📋 Copied original file instead")
                except Exception as copy_error:
                    print(f"  ✗ Failed to copy original: {copy_error}")

    def translate_documents(self):
        """Main translation method with total runtime tracking."""
        print("🚀 Starting improved medical document translation...")
        print("📋 Translation Strategy:")
        print("   • Simplified medical term preservation with whitelist")
        print("   • Improved chunking with larger sizes and better boundaries")
        print("   • Enhanced quality assessment with placeholder detection")
        print("   • facebook/nllb-200-3.3B as primary model with fallbacks")
        print("   • Better English quality validation")

        # Start total runtime tracking
        self.total_translation_start_time = datetime.now()
        print(f"🕐 Translation started at: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Find files
        json_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, self.input_dir)
                    output_path = os.path.join(self.output_dir, rel_path)
                    json_files.append((input_path, output_path))

        total_files = len(json_files)
        print(f"📁 Found {total_files} JSON files to process")

        if total_files == 0:
            print("⚠️  No JSON files found in input directory")
            return

        # Group files by directory for batch processing
        file_groups = {}
        for input_path, output_path in json_files:
            dir_key = os.path.basename(os.path.dirname(input_path))
            if dir_key not in file_groups:
                file_groups[dir_key] = []
            file_groups[dir_key].append((input_path, output_path))

        # Process groups
        for group_dir, group_files in file_groups.items():
            print(f"\n📦 Processing batch from {group_dir}")
            self.process_batch_files(group_files)

        # Calculate total runtime
        total_translation_end_time = datetime.now()
        total_runtime_seconds = (total_translation_end_time - self.total_translation_start_time).total_seconds()
        total_runtime_minutes = total_runtime_seconds / 60
        total_runtime_hours = total_runtime_minutes / 60

        # Summary with runtime information
        print(f"\n🎉 Improved Medical Translation Complete!")
        print(f"📊 Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")
        print(f"   • Simplified medical term preservation used")
        print(f"   • Improved chunking and quality assessment applied")
        print(f"\n⏱️  Runtime Summary:")
        print(f"   • Start time: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • End time: {total_translation_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Total runtime: {total_runtime_seconds:.2f} seconds")
        print(f"   • Total runtime: {total_runtime_minutes:.2f} minutes")
        if total_runtime_hours >= 1:
            print(f"   • Total runtime: {total_runtime_hours:.2f} hours")
        if total_files > 0:
            avg_time_per_file = total_runtime_seconds / total_files
            print(f"   • Average time per file: {avg_time_per_file:.2f} seconds")