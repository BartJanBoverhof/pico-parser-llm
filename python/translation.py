class Translator:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0
        self.total_translation_start_time = None
        self.processing_start_time = None

        self.model_name = 'facebook/nllb-200-3.3B'

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

        self.quality_thresholds = {
            'minimum_acceptable_quality': 0.40,
            'length_ratio_threshold': 0.60,
            'minimum_length_ratio': 0.30,
            'content_preservation_threshold': 0.70,
            'sentence_preservation_threshold': 0.80,
        }

        self.max_input_tokens = 480
        self.chunk_params = {
            'target_tokens': 300,
            'max_tokens': 380,
            'overlap_tokens': 40,
            'min_tokens': 40,
            'overlap_chars': 120,
        }

        self.table_patterns = [
            r'(?:Row|Column)\s+\d+[:\s]',
            r'Table\s+(?:Title|contains|shows)',
            r'\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|',
            r'€\s*\d+(?:[.,]\d+)*',
            r'\d+(?:[.,]\d+)*\s*€',
        ]

        self.domain_patterns = {
            'medical_terms': [
                r'\b(?:mg|kg|ml|mcg|µg|IU|units?)\b',
                r'\b(?:patient|treatment|therapy|drug|medication|dose|dosage)\b',
                r'\b(?:clinical|trial|study|analysis|outcome|endpoint)\b',
                r'\b(?:efficacy|safety|adverse|effect|reaction|event)\b'
            ],
            'statistical_terms': [
                r'\b(?:hazard ratio|odds ratio|confidence interval|p-value|p value)\b',
                r'\b(?:significant|significance|statistical|analysis|regression)\b',
                r'\b(?:median|mean|average|standard deviation|variance)\b',
                r'\b(?:correlation|association|relationship|comparison)\b'
            ],
            'numerical_patterns': [
                r'\b\d+(?:[.,]\d+)*\s*(?:mg|kg|ml|mcg|µg|IU|units?|%|percent)\b',
                r'\b\d+(?:[.,]\d+)*\s*(?:months?|weeks?|days?|years?|hours?)\b',
                r'\b\d+(?:[.,]\d+)*(?:\s*[-–]\s*\d+(?:[.,]\d+)*)?%?\b'
            ]
        }

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

        self.current_translator = None
        self.current_tokenizer = None
        self.current_language = None
        self.current_document_language = None

    def count_tokens_accurately(self, text: str) -> int:
        if not text:
            return 0

        if self.current_tokenizer:
            try:
                tokens = self.current_tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception:
                pass

        return max(1, len(text) // 4)

    def split_text_into_chunks(self, text: str) -> List[str]:
        if not text:
            return []

        total_tokens = self.count_tokens_accurately(text)
        target_tokens = self.chunk_params['target_tokens']
        max_tokens = self.chunk_params['max_tokens']
        overlap_tokens = self.chunk_params['overlap_tokens']
        min_tokens = self.chunk_params['min_tokens']

        if total_tokens <= max_tokens:
            return [text]

        chunks = []
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [text]

        current_chunk = ""
        current_tokens = 0
        sentence_index = 0

        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            sentence_tokens = self.count_tokens_accurately(sentence)

            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                word_chunks = self._split_large_sentence(sentence, max_tokens)
                chunks.extend(word_chunks[:-1])

                if word_chunks:
                    current_chunk = word_chunks[-1]
                    current_tokens = self.count_tokens_accurately(current_chunk)

                sentence_index += 1
                continue

            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunks.append(current_chunk.strip())

                overlap_chunk = self._create_overlap_chunk(sentences, sentence_index, overlap_tokens)
                current_chunk = overlap_chunk + (" " + sentence if overlap_chunk else sentence)
                current_tokens = self.count_tokens_accurately(current_chunk)
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
                current_tokens += sentence_tokens

            sentence_index += 1

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        filtered_chunks = [chunk for chunk in chunks if self.count_tokens_accurately(chunk) >= min_tokens]
        return filtered_chunks if filtered_chunks else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        final_sentences = []
        for sentence in sentences:
            if '\n\n' in sentence:
                parts = sentence.split('\n\n')
                final_sentences.extend([part.strip() for part in parts if part.strip()])
            else:
                final_sentences.append(sentence.strip())

        return [s for s in final_sentences if s]

    def _split_large_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        words = sentence.split()
        chunks = []
        current_chunk = []

        for word in words:
            test_chunk = ' '.join(current_chunk + [word])
            if self.count_tokens_accurately(test_chunk) > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [sentence]

    def _create_overlap_chunk(self, sentences: List[str], current_index: int, overlap_tokens: int) -> str:
        if current_index == 0 or overlap_tokens <= 0:
            return ""

        overlap_text = ""
        tokens_used = 0

        for i in range(current_index - 1, -1, -1):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens_accurately(sentence)

            if tokens_used + sentence_tokens <= overlap_tokens:
                overlap_text = sentence + (" " + overlap_text if overlap_text else "")
                tokens_used += sentence_tokens
            else:
                if tokens_used == 0:
                    words = sentence.split()
                    partial_sentence = ""
                    for word in reversed(words):
                        test_text = word + (" " + partial_sentence if partial_sentence else "")
                        if self.count_tokens_accurately(test_text) <= overlap_tokens:
                            partial_sentence = test_text
                        else:
                            break
                    if partial_sentence:
                        overlap_text = partial_sentence
                break

        return overlap_text

    def adaptive_chunk_for_translation(self, text: str) -> List[str]:
        if not text:
            return []

        text_tokens = self.count_tokens_accurately(text)
        max_tokens = self.chunk_params['max_tokens']

        if text_tokens <= max_tokens:
            return [text]

        print(f"      🔨 Chunking needed ({text_tokens} tokens > {max_tokens} limit)")

        chunks = self.split_text_into_chunks(text)

        oversized_chunks = 0
        for chunk in chunks:
            chunk_tokens = self.count_tokens_accurately(chunk)
            if chunk_tokens > self.max_input_tokens:
                oversized_chunks += 1

        if oversized_chunks > 0:
            print(f"      ⚠️  {oversized_chunks} chunks exceed input token limit")

        print(f"      ✓ Created {len(chunks)} chunks with conservative overlap")
        return chunks

    def is_table_content(self, text: str) -> bool:
        if not text:
            return False

        for pattern in self.table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        line_count = len(text.split('\n'))
        pipe_density = text.count('|') / max(len(text), 1)
        numeric_density = len(re.findall(r'\d+(?:\.\d+)?', text)) / max(len(text.split()), 1)

        return (line_count >= 5 and pipe_density > 0.05) or numeric_density > 0.3

    def extract_domain_elements(self, text: str) -> Dict[str, List[str]]:
        elements = {}

        for category, patterns in self.domain_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            elements[category] = list(set(matches))

        return elements

    def calculate_term_consistency(self, original_terms: List[str], translated_terms: List[str]) -> float:
        if not original_terms and not translated_terms:
            return 1.0
        if not original_terms or not translated_terms:
            return 0.0

        orig_counts = Counter(original_terms)
        trans_counts = Counter(translated_terms)

        consistency_scores = []
        for term in orig_counts:
            if term in trans_counts:
                expected_count = orig_counts[term]
                actual_count = trans_counts[term]
                consistency = min(actual_count, expected_count) / max(actual_count, expected_count)
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(0.0)

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

    def assess_linguistic_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        if not translated_text.strip():
            return {
                'fluency': 0.0,
                'accuracy': 0.0,
                'consistency': 0.0,
                'completeness': 0.0,
                'linguistic_composite': 0.0
            }

        fluency_score = self.assess_fluency(translated_text)
        accuracy_score = self.assess_accuracy(original_text, translated_text, language)
        consistency_score = self.assess_consistency(original_text, translated_text)
        completeness_score = self.assess_completeness(original_text, translated_text)

        linguistic_composite = (fluency_score * 0.3 + accuracy_score * 0.3 +
                               consistency_score * 0.2 + completeness_score * 0.2)

        return {
            'fluency': fluency_score,
            'accuracy': accuracy_score,
            'consistency': consistency_score,
            'completeness': completeness_score,
            'linguistic_composite': linguistic_composite
        }

    def assess_fluency(self, text: str) -> float:
        if not text.strip():
            return 0.0

        fluency_score = 1.0

        try:
            detected_lang = detect(text)
            if detected_lang != 'en':
                fluency_score *= 0.3
        except:
            fluency_score *= 0.5

        english_markers = [
            r'\b(?:the|and|of|to|a|in|is|it|that|for|on|with|as|be|at|by|this|have|from|or|one|had|but|not|what|all|were|they|we|when|your|can|said|there|use|an|each|which|she|do|how|their|if|will|up|other|about|out|many|then|them|would|so|some|her|him|into|has|more|two|go|no|way|could|my|than|first|been|who|oil|its|now|find|long|down|day|did|get|make|may)\b',
            r'[.!?]',
            r'\b[A-Z][a-z]+\b'
        ]

        for pattern in english_markers:
            matches = len(re.findall(pattern, text))
            if matches == 0:
                fluency_score *= 0.8

        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 3)
        sentence_ratio = complete_sentences / max(len([s for s in sentences if s.strip()]), 1)
        fluency_score *= sentence_ratio

        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                fluency_score *= 0.6

        return max(0.0, min(1.0, fluency_score))

    def assess_accuracy(self, original_text: str, translated_text: str, language: str) -> float:
        if not original_text.strip() or not translated_text.strip():
            return 0.0

        orig_len = len(original_text.strip())
        trans_len = len(translated_text.strip())
        length_ratio = trans_len / max(orig_len, 1)

        if length_ratio < 0.3:
            length_score = 0.1
        elif length_ratio < 0.6:
            length_score = 0.6
        elif length_ratio < 1.5:
            length_score = 1.0
        else:
            length_score = 0.8

        orig_sentences = self._split_into_sentences(original_text)
        trans_sentences = self._split_into_sentences(translated_text)
        sentence_ratio = len(trans_sentences) / max(len(orig_sentences), 1)
        sentence_score = min(1.0, sentence_ratio) if sentence_ratio > 0.7 else sentence_ratio * 0.8

        semantic_score = self.assess_semantic_preservation(original_text, translated_text)

        accuracy_composite = (length_score * 0.3 + sentence_score * 0.3 + semantic_score * 0.4)
        return max(0.0, min(1.0, accuracy_composite))

    def assess_semantic_preservation(self, original_text: str, translated_text: str) -> float:
        orig_words = set(re.findall(r'\b\w+\b', original_text.lower()))
        trans_words = set(re.findall(r'\b\w+\b', translated_text.lower()))

        if not orig_words:
            return 1.0 if not trans_words else 0.0
        if not trans_words:
            return 0.0

        overlap = len(orig_words.intersection(trans_words))
        union = len(orig_words.union(trans_words))
        jaccard_similarity = overlap / union if union > 0 else 0.0

        return min(1.0, jaccard_similarity * 2)

    def assess_consistency(self, original_text: str, translated_text: str) -> float:
        orig_elements = self.extract_domain_elements(original_text)
        trans_elements = self.extract_domain_elements(translated_text)

        consistency_scores = []

        for category in orig_elements:
            if category in trans_elements:
                score = self.calculate_term_consistency(
                    orig_elements[category],
                    trans_elements[category]
                )
                consistency_scores.append(score)
            else:
                consistency_scores.append(0.0 if orig_elements[category] else 1.0)

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    def assess_completeness(self, original_text: str, translated_text: str) -> float:
        orig_paras = [p.strip() for p in original_text.split('\n\n') if p.strip()]
        trans_paras = [p.strip() for p in translated_text.split('\n\n') if p.strip()]

        para_ratio = len(trans_paras) / max(len(orig_paras), 1)
        para_score = min(1.0, para_ratio) if para_ratio > 0.8 else para_ratio * 0.9

        orig_words = len(original_text.split())
        trans_words = len(translated_text.split())
        word_ratio = trans_words / max(orig_words, 1)
        word_score = min(1.0, word_ratio) if word_ratio > 0.7 else word_ratio * 0.8

        return (para_score + word_score) / 2

    def assess_domain_specific_quality(self, original_text: str, translated_text: str) -> Dict[str, float]:
        medical_score = self.assess_medical_terminology(original_text, translated_text)
        numerical_score = self.assess_numerical_integrity(original_text, translated_text)
        statistical_score = self.assess_statistical_terms(original_text, translated_text)
        unit_score = self.assess_unit_preservation(original_text, translated_text)

        domain_composite = (medical_score * 0.3 + numerical_score * 0.3 +
                           statistical_score * 0.2 + unit_score * 0.2)

        return {
            'medical_terminology': medical_score,
            'numerical_integrity': numerical_score,
            'statistical_terms': statistical_score,
            'unit_preservation': unit_score,
            'domain_composite': domain_composite
        }

    def assess_medical_terminology(self, original_text: str, translated_text: str) -> float:
        medical_patterns = self.domain_patterns['medical_terms']

        orig_terms = []
        trans_terms = []

        for pattern in medical_patterns:
            orig_terms.extend(re.findall(pattern, original_text, re.IGNORECASE))
            trans_terms.extend(re.findall(pattern, translated_text, re.IGNORECASE))

        if not orig_terms:
            return 1.0

        orig_unique = set([term.lower() for term in orig_terms])
        trans_unique = set([term.lower() for term in trans_terms])

        preserved_ratio = len(orig_unique.intersection(trans_unique)) / len(orig_unique)
        return min(1.0, preserved_ratio * 1.2)

    def assess_numerical_integrity(self, original_text: str, translated_text: str) -> float:
        number_pattern = r'\b\d+(?:[.,]\d+)*\b'

        orig_numbers = re.findall(number_pattern, original_text)
        trans_numbers = re.findall(number_pattern, translated_text)

        if not orig_numbers:
            return 1.0 if not trans_numbers else 0.9

        orig_normalized = [num.replace(',', '.') for num in orig_numbers]
        trans_normalized = [num.replace(',', '.') for num in trans_numbers]

        orig_set = set(orig_normalized)
        trans_set = set(trans_normalized)

        preserved_numbers = len(orig_set.intersection(trans_set))
        preservation_ratio = preserved_numbers / len(orig_set)

        return preservation_ratio

    def assess_statistical_terms(self, original_text: str, translated_text: str) -> float:
        stat_patterns = self.domain_patterns['statistical_terms']

        orig_stats = []
        trans_stats = []

        for pattern in stat_patterns:
            orig_stats.extend(re.findall(pattern, original_text, re.IGNORECASE))
            trans_stats.extend(re.findall(pattern, translated_text, re.IGNORECASE))

        if not orig_stats:
            return 1.0

        return self.calculate_term_consistency(orig_stats, trans_stats)

    def assess_unit_preservation(self, original_text: str, translated_text: str) -> float:
        unit_pattern = r'\b\d+(?:[.,]\d+)*\s*(?:mg|kg|ml|mcg|µg|IU|units?|%|percent|months?|weeks?|days?|years?|hours?)\b'

        orig_units = re.findall(unit_pattern, original_text, re.IGNORECASE)
        trans_units = re.findall(unit_pattern, translated_text, re.IGNORECASE)

        if not orig_units:
            return 1.0

        orig_normalized = [unit.lower().replace(' ', '') for unit in orig_units]
        trans_normalized = [unit.lower().replace(' ', '') for unit in trans_units]

        preserved = len(set(orig_normalized).intersection(set(trans_normalized)))
        return preserved / len(set(orig_normalized))

    def assess_structural_quality(self, original_data: dict, translated_data: dict) -> Dict[str, float]:
        format_score = self.assess_format_preservation(original_data, translated_data)
        integrity_score = self.assess_document_integrity(original_data, translated_data)
        architecture_score = self.assess_information_architecture(original_data, translated_data)

        structural_composite = (format_score * 0.4 + integrity_score * 0.4 + architecture_score * 0.2)

        return {
            'format_preservation': format_score,
            'document_integrity': integrity_score,
            'information_architecture': architecture_score,
            'structural_composite': structural_composite
        }

    def assess_format_preservation(self, original_data: dict, translated_data: dict) -> float:
        if 'chunks' not in original_data or 'chunks' not in translated_data:
            return 0.0

        orig_chunks = original_data['chunks']
        trans_chunks = translated_data['chunks']

        if len(orig_chunks) != len(trans_chunks):
            return 0.5

        format_scores = []
        for orig, trans in zip(orig_chunks, trans_chunks):
            chunk_score = 1.0

            if 'heading' in orig and bool(orig['heading']) != bool(trans.get('heading', '')):
                chunk_score *= 0.7

            if 'text' in orig:
                orig_is_table = self.is_table_content(orig['text'])
                trans_is_table = self.is_table_content(trans.get('text', ''))
                if orig_is_table != trans_is_table:
                    chunk_score *= 0.8

            format_scores.append(chunk_score)

        return sum(format_scores) / len(format_scores) if format_scores else 0.0

    def assess_document_integrity(self, original_data: dict, translated_data: dict) -> float:
        integrity_score = 1.0

        orig_keys = set(original_data.keys())
        trans_keys = set(translated_data.keys())

        if '_translation_metadata' in trans_keys:
            trans_keys.remove('_translation_metadata')

        key_preservation = len(orig_keys.intersection(trans_keys)) / len(orig_keys) if orig_keys else 1.0
        integrity_score *= key_preservation

        if 'chunks' in original_data and 'chunks' in translated_data:
            chunk_ratio = len(translated_data['chunks']) / max(len(original_data['chunks']), 1)
            if chunk_ratio < 0.9:
                integrity_score *= 0.7

        return integrity_score

    def assess_information_architecture(self, original_data: dict, translated_data: dict) -> float:
        if 'chunks' not in original_data or 'chunks' not in translated_data:
            return 0.5

        orig_structure = []
        trans_structure = []

        for chunk in original_data['chunks']:
            has_heading = bool(chunk.get('heading', ''))
            has_text = bool(chunk.get('text', ''))
            orig_structure.append((has_heading, has_text))

        for chunk in translated_data['chunks']:
            has_heading = bool(chunk.get('heading', ''))
            has_text = bool(chunk.get('text', ''))
            trans_structure.append((has_heading, has_text))

        if len(orig_structure) != len(trans_structure):
            return 0.6

        matching_structure = sum(1 for o, t in zip(orig_structure, trans_structure) if o == t)
        return matching_structure / len(orig_structure) if orig_structure else 1.0

    def assess_content_preservation(self, original_text: str, translated_text: str) -> Dict[str, float]:
        if not original_text.strip() or not translated_text.strip():
            return {'content_preservation': 0.0, 'sentence_preservation': 0.0, 'paragraph_preservation': 0.0}

        original_sentences = self._split_into_sentences(original_text)
        translated_sentences = self._split_into_sentences(translated_text)

        original_paragraphs = [p.strip() for p in original_text.split('\n\n') if p.strip()]
        translated_paragraphs = [p.strip() for p in translated_text.split('\n\n') if p.strip()]

        sentence_ratio = len(translated_sentences) / max(len(original_sentences), 1)
        sentence_preservation = min(1.0, sentence_ratio) if sentence_ratio > 0.5 else sentence_ratio * 0.5

        paragraph_ratio = len(translated_paragraphs) / max(len(original_paragraphs), 1)
        paragraph_preservation = min(1.0, paragraph_ratio) if paragraph_ratio > 0.5 else paragraph_ratio * 0.5

        original_words = len(original_text.split())
        translated_words = len(translated_text.split())
        word_ratio = translated_words / max(original_words, 1)

        if word_ratio < 0.4:
            content_preservation = word_ratio * 0.5
        elif word_ratio < 0.7:
            content_preservation = word_ratio * 0.8
        else:
            content_preservation = min(1.0, word_ratio)

        return {
            'content_preservation': content_preservation,
            'sentence_preservation': sentence_preservation,
            'paragraph_preservation': paragraph_preservation,
            'word_ratio': word_ratio,
            'sentence_ratio': sentence_ratio,
            'paragraph_ratio': paragraph_ratio
        }

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        if not translated_text.strip():
            return {
                'overall': 0.0,
                'linguistic': self.assess_linguistic_quality(original_text, translated_text, language),
                'domain_specific': self.assess_domain_specific_quality(original_text, translated_text),
                'legacy_scores': {'english_quality': 0.0, 'length_score': 0.0, 'content_preservation': 0.0}
            }

        linguistic_scores = self.assess_linguistic_quality(original_text, translated_text, language)
        domain_scores = self.assess_domain_specific_quality(original_text, translated_text)

        legacy_content = self.assess_content_preservation(original_text, translated_text)
        legacy_scores = {
            'english_quality': linguistic_scores['fluency'],
            'length_score': linguistic_scores['completeness'],
            'content_preservation': legacy_content['content_preservation'],
            'length_ratio': legacy_content['word_ratio']
        }
        legacy_scores.update(legacy_content)

        overall_score = (linguistic_scores['linguistic_composite'] * 0.6 +
                        domain_scores['domain_composite'] * 0.4)

        return {
            'overall': overall_score,
            'linguistic': linguistic_scores,
            'domain_specific': domain_scores,
            'legacy_scores': legacy_scores
        }

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        if 'chunks' not in translated_data or 'chunks' not in original_data:
            return {
                'overall': 0.0,
                'chunk_count': 0,
                'linguistic': {'linguistic_composite': 0.0},
                'domain_specific': {'domain_composite': 0.0},
                'structural': {'structural_composite': 0.0}
            }

        chunk_assessments = []
        missing_chunks = 0
        empty_translations = 0

        for i, (orig_chunk, trans_chunk) in enumerate(zip(original_data['chunks'], translated_data['chunks'])):
            if 'text' in orig_chunk and 'text' in trans_chunk:
                if not orig_chunk['text']:
                    continue

                if not trans_chunk['text']:
                    empty_translations += 1
                    chunk_assessments.append({
                        'overall': 0.0,
                        'linguistic': {'linguistic_composite': 0.0},
                        'domain_specific': {'domain_composite': 0.0}
                    })
                else:
                    assessment = self.assess_translation_quality(
                        orig_chunk['text'], trans_chunk['text'], language
                    )
                    chunk_assessments.append(assessment)

        structural_scores = self.assess_structural_quality(original_data, translated_data)

        if not chunk_assessments:
            return {
                'overall': 0.0,
                'chunk_count': 0,
                'linguistic': {'linguistic_composite': 0.0},
                'domain_specific': {'domain_composite': 0.0},
                'structural': structural_scores,
                'empty_translations': 0,
                'missing_content_ratio': 0.0
            }

        avg_overall = sum(a['overall'] for a in chunk_assessments) / len(chunk_assessments)

        linguistic_keys = ['fluency', 'accuracy', 'consistency', 'completeness', 'linguistic_composite']
        avg_linguistic = {}
        for key in linguistic_keys:
            scores = [a['linguistic'][key] for a in chunk_assessments if key in a['linguistic']]
            avg_linguistic[key] = sum(scores) / len(scores) if scores else 0.0

        domain_keys = ['medical_terminology', 'numerical_integrity', 'statistical_terms', 'unit_preservation', 'domain_composite']
        avg_domain = {}
        for key in domain_keys:
            scores = [a['domain_specific'][key] for a in chunk_assessments if key in a['domain_specific']]
            avg_domain[key] = sum(scores) / len(scores) if scores else 0.0

        missing_content_ratio = empty_translations / max(len(chunk_assessments), 1)

        if missing_content_ratio > 0.1:
            avg_overall *= (1.0 - missing_content_ratio)
            avg_linguistic['linguistic_composite'] *= (1.0 - missing_content_ratio)
            avg_domain['domain_composite'] *= (1.0 - missing_content_ratio)

        final_overall = (avg_overall * 0.7 + structural_scores['structural_composite'] * 0.3)

        return {
            'overall': final_overall,
            'chunk_count': len(chunk_assessments),
            'linguistic': avg_linguistic,
            'domain_specific': avg_domain,
            'structural': structural_scores,
            'empty_translations': empty_translations,
            'missing_content_ratio': missing_content_ratio
        }

    def detect_document_language(self, text: str) -> Optional[str]:
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
        if not text or not text.strip():
            return True

        t = text.strip()

        alpha_chars = re.findall(r'[A-Za-z]', t)
        if len(alpha_chars) < 2:
            return True

        try:
            try:
                from langdetect import detect_langs as _detect_langs
            except Exception:
                _detect_langs = None

            if _detect_langs is not None:
                langs = _detect_langs(t)
                en_prob = next((l.prob for l in langs if l.lang == 'en'), 0.0)
                doc_lang = self.current_document_language
                doc_prob = next((l.prob for l in langs if l.lang == doc_lang), 0.0) if doc_lang else 0.0

                if doc_lang and doc_lang != 'en':
                    return en_prob >= 0.90 and en_prob >= doc_prob + 0.20
                else:
                    return en_prob >= 0.85
            else:
                lang = detect(t)
                if self.current_document_language and self.current_document_language != 'en':
                    return lang == 'en'
                return lang == 'en'
        except Exception:
            if self.current_document_language and self.current_document_language != 'en':
                return False
            return True

    def check_model_availability(self, model_name: str) -> bool:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"    ✗ Model {model_name} not available: {str(e)[:50]}")
            return False

    def load_nllb_model(self, model_name: str, language: str) -> Optional[Any]:
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

            self.current_tokenizer = tokenizer

            src_lang_code = self.nllb_lang_mapping.get(language)
            tgt_lang_code = self.nllb_lang_mapping.get('en', 'eng_Latn')
            if src_lang_code:
                try:
                    tokenizer.src_lang = src_lang_code
                except Exception:
                    pass

            def nllb_translate(text, generation_params=None):
                try:
                    max_input_length = self.max_input_tokens

                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    input_length = inputs['input_ids'].shape[1]
                    output_max_length = min(512, max(200, int(input_length * 1.8)))

                    gen_kwargs = {
                        'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang_code),
                        'max_length': output_max_length,
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
        if self.current_language == language and self.current_translator:
            return self.current_translator

        self.clear_translator()

        print(f"  🔄 Loading translator for language: {language}")

        if language not in self.nllb_lang_mapping:
            print(f"    ✗ Language {language} not supported by NLLB")
            return None

        if not self.check_model_availability(self.model_name):
            print(f"    ✗ Model {self.model_name} not available")
            return None

        translator = self.load_nllb_model(self.model_name, language)

        if translator:
            self.current_translator = translator
            self.current_language = language
            print(f"    ✓ Successfully loaded translator for {language}")
            return translator
        else:
            print(f"    ✗ No translator available for language: {language}")
            return None

    def translate_single_chunk(self, text: str, translator) -> str:
        if not text.strip() or not translator:
            return text

        try:
            chunks = self.adaptive_chunk_for_translation(text)

            if len(chunks) == 1:
                chunk_tokens = self.count_tokens_accurately(text)
                gen_params = {
                    'max_length': min(512, max(200, int(chunk_tokens * 1.8))),
                    'truncation': True,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 1.1,
                    'do_sample': False,
                    'num_beams': 4,
                }

                result = translator(text, generation_params=gen_params)
                translated_text = result[0]['translation_text']
            else:
                translated_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_tokens = self.count_tokens_accurately(chunk)
                    gen_params = {
                        'max_length': min(512, max(200, int(chunk_tokens * 1.8))),
                        'truncation': True,
                        'no_repeat_ngram_size': 3,
                        'repetition_penalty': 1.1,
                        'do_sample': False,
                        'num_beams': 4,
                    }

                    result = translator(chunk, generation_params=gen_params)
                    translated_chunks.append(result[0]['translation_text'])

                translated_text = self._merge_translated_chunks(translated_chunks)
                print(f"      ✓ Merged {len(translated_chunks)} chunks")

            return translated_text.strip()

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def _merge_translated_chunks(self, chunks: List[str]) -> str:
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]

        merged = chunks[0]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]

            words_merged = merged.split()
            words_current = current_chunk.split()

            overlap_found = False
            for overlap_size in range(min(8, len(words_merged), len(words_current)), 0, -1):
                if words_merged[-overlap_size:] == words_current[:overlap_size]:
                    merged += " " + " ".join(words_current[overlap_size:])
                    overlap_found = True
                    break

            if not overlap_found:
                merged += " " + current_chunk

        return merged

    def translate_document(self, data: dict, language: str) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        print(f"  📝 Translating document")

        translation_start_time = datetime.now()
        self.current_document_language = language

        translator = self.load_translator_for_language(language)
        if not translator:
            print(f"    ✗ No translator available")
            self.current_document_language = None
            return data, {'overall': 0.0}, {
                'model_loaded': False,
                'processing_time_seconds': 0,
                'model_name': None
            }

        translated_data = copy.deepcopy(data)

        if 'chunks' not in translated_data:
            print(f"    No chunks found in document")
            self.current_document_language = None
            return translated_data, {'overall': 0.0}, {
                'model_loaded': True,
                'processing_time_seconds': (datetime.now() - translation_start_time).total_seconds(),
                'model_name': self.model_name,
                'chunks_found': False
            }

        total_chunks = len(translated_data['chunks'])
        translated_count = 0
        english_count = 0
        table_count = 0

        print(f"    Processing {total_chunks} chunks with conservative chunking...")

        for i, chunk in enumerate(translated_data['chunks']):
            if i % 20 == 0 or i == total_chunks - 1:
                print(f"      Chunk {i+1}/{total_chunks}")

            if 'heading' in chunk and chunk['heading']:
                if self.is_english_chunk(chunk['heading']):
                    english_count += 1
                else:
                    chunk['heading'] = self.translate_single_chunk(
                        chunk['heading'], translator
                    )
                    translated_count += 1

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

        quality_scores = self.assess_document_quality(translated_data, data, language)

        print(f"    📊 Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      Linguistic Composite: {quality_scores['linguistic']['linguistic_composite']:.3f}")
        print(f"      Domain Composite: {quality_scores['domain_specific']['domain_composite']:.3f}")
        print(f"      Structural Composite: {quality_scores['structural']['structural_composite']:.3f}")
        print(f"      Missing Content Ratio: {quality_scores.get('missing_content_ratio', 0):.3f}")

        translation_metadata = {
            'model_loaded': True,
            'model_name': self.model_name,
            'processing_time_seconds': processing_time,
            'chunks_found': True,
            'total_chunks': total_chunks,
            'chunks_translated': translated_count,
            'chunks_english': english_count,
            'table_chunks_processed': table_count,
            'quality_scores': quality_scores
        }

        self.current_document_language = None
        return translated_data, quality_scores, translation_metadata

    def clear_translator(self):
        self.current_translator = None
        self.current_tokenizer = None
        self.current_language = None

        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def process_json_file(self, input_path: str, output_path: str):
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")

        self.processing_start_time = datetime.now()

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading JSON: {e}")
            return

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

        base_metadata = {
            "processing_timestamp": self.processing_start_time.isoformat(),
            "source_file": file_name,
            "detected_language": document_language,
            "was_translation_needed": False,
            "translation_strategy": "nllb_translation",
            "max_input_tokens": self.max_input_tokens,
            "target_chunk_tokens": self.chunk_params['target_tokens'],
            "overlap_tokens": self.chunk_params['overlap_tokens'],
            "table_content_detected": sum(1 for chunk in data.get('chunks', [])
                                        if self.is_table_content(chunk.get('text', '')))
        }

        if not document_language or document_language == 'en':
            print(f"  📋 Document is English, copying to output directory")

            translation_metadata = copy.deepcopy(base_metadata)
            translation_metadata.update({
                "translation_decision": "no_translation_needed_english",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })

            data_copy = copy.deepcopy(data)
            data_copy["_translation_metadata"] = translation_metadata

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_copy, f, indent=2, ensure_ascii=False)
            return

        model_available = document_language in self.nllb_lang_mapping and self.check_model_availability(self.model_name)

        base_metadata.update({
            "was_translation_needed": True,
            "model_available": model_available
        })

        if not model_available:
            print(f"  📋 No translation model available for language {document_language}")

            translation_metadata = copy.deepcopy(base_metadata)
            translation_metadata.update({
                "translation_decision": "no_model_available",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })

            data_copy = copy.deepcopy(data)
            data_copy["_translation_metadata"] = translation_metadata

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_copy, f, indent=2, ensure_ascii=False)
            return

        print(f"\n  🎯 Processing with NLLB model")
        translated_data, quality_scores, translation_metadata_inner = self.translate_document(
            data, document_language
        )

        self.clear_translator()

        total_processing_time = (datetime.now() - self.processing_start_time).total_seconds()

        translation_metadata = copy.deepcopy(base_metadata)
        translation_metadata.update({
            "translation_decision": "nllb_processing",
            "model_used": self.model_name,
            "quality_scores": quality_scores,
            "translation_metadata": translation_metadata_inner,
            "total_processing_time_seconds": total_processing_time,
            "processing_completed_timestamp": datetime.now().isoformat()
        })

        final_data = translated_data
        final_data["_translation_metadata"] = translation_metadata

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved: Overall {quality_scores['overall']:.3f}, Linguistic {quality_scores['linguistic']['linguistic_composite']:.3f}, Domain {quality_scores['domain_specific']['domain_composite']:.3f}")

        if 'chunks_english' in translation_metadata_inner:
            self.english_chunks_preserved += translation_metadata_inner['chunks_english']
        if 'chunks_translated' in translation_metadata_inner:
            self.chunks_translated += translation_metadata_inner['chunks_translated']

        print(f"  ⏱️  Total processing time: {total_processing_time:.2f}s")

    def translate_documents(self):
        print("🚀 Starting document translation...")
        print("📋 Strategy:")
        print("   • Process documents with NLLB-200-3.3B model")
        print("   • Multi-dimensional quality assessment (Linguistic, Domain, Structural)")
        print("   • Conservative chunking with improved content preservation")
        print("   • Enhanced validation to detect missing translations")
        print(f"   • Target chunk size: {self.chunk_params['target_tokens']} tokens")
        print(f"   • Max chunk size: {self.chunk_params['max_tokens']} tokens")

        self.total_translation_start_time = datetime.now()
        print(f"🕐 Translation started at: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        os.makedirs(self.output_dir, exist_ok=True)

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

        for input_path, output_path in json_files:
            try:
                self.process_json_file(input_path, output_path)
            except Exception as e:
                print(f"  ✗ Processing error for {os.path.basename(input_path)}: {str(e)[:100]}")
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(input_path, output_path)
                    print(f"  📋 Copied original file to output directory")
                except Exception as copy_error:
                    print(f"  ✗ Failed to copy original: {copy_error}")

        total_translation_end_time = datetime.now()
        total_runtime_seconds = (total_translation_end_time - self.total_translation_start_time).total_seconds()
        total_runtime_minutes = total_runtime_seconds / 60
        total_runtime_hours = total_runtime_minutes / 60

        print(f"\n🎉 Translation Quality Assessment Complete!")
        print(f"📊 Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")
        print(f"   • Results saved to: {self.output_dir}")
        print(f"   • Quality dimensions: Linguistic, Domain-Specific, Structural")
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