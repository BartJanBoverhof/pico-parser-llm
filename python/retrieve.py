import os
import json
import re
import tiktoken
from datetime import datetime
from typing import List, Dict, Any, Optional


class TextSimilarityUtils:
    """
    Utility class for text similarity and comparator extraction functions.
    """
    @staticmethod
    def jaccard_similarity(text1, text2):
        """Calculate Jaccard similarity between two text strings."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        
        return intersection / union if union > 0 else 0

    @staticmethod
    def is_subset(text1, text2):
        """Check if text1 is effectively contained within text2."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        overlap_ratio = len(tokens1.intersection(tokens2)) / len(tokens1) if tokens1 else 0
        return overlap_ratio > 0.9

    @staticmethod
    def extract_potential_comparators(text):
        """
        Extract potential drug names/comparators from text using pattern matching.
        """
        words = text.split()
        capitalized_words = []
        
        for i, word in enumerate(words):
            if (i > 0 and words[i-1][-1] in '.!?') or i == 0:
                clean_word = word.strip('.,;:()[]{}')
                if clean_word and clean_word[0].isupper() and len(clean_word) > 1 and clean_word.lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'by', 'to', 'of']:
                    capitalized_words.append(clean_word)
        
        dosage_pattern = r'\b\w+\s+\d+\s*(?:mg|mcg|g|ml)\b'
        dosages = re.findall(dosage_pattern, text)
        
        suffix_pattern = r'\b\w+(?:mab|nib|zumab|tinib|ciclib|parib|vastatin)\b'
        suffix_matches = re.findall(suffix_pattern, text.lower())
        
        all_matches = capitalized_words + dosages + suffix_matches
        
        common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'not', 'are', 'from', 'was', 'were'}
        filtered_matches = [m for m in all_matches if m.lower() not in common_words]
        
        return set(filtered_matches)


class DocumentDeduplicator:
    """
    Class to handle deduplication of retrieved documents and context optimization.
    """
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.similarity_utils = TextSimilarityUtils()
    
    def deduplicate_documents(self, docs, preserve_country_diversity=True):
        """
        Deduplicate similar documents while preserving diversity.
        """
        unique_docs = []
        seen_texts = set()
        removed_docs = []
        
        for doc in docs:
            text = doc.page_content.strip()
            if text in seen_texts:
                removed_docs.append({
                    "doc": doc,
                    "reason": "exact duplicate",
                    "similar_to": None
                })
                continue
                
            is_duplicate = False
            similar_to = None
            
            for kept_doc in unique_docs:
                if preserve_country_diversity and doc.metadata.get("country") != kept_doc.metadata.get("country"):
                    continue
                    
                similarity = self.similarity_utils.jaccard_similarity(text, kept_doc.page_content)
                is_subset_relation = (self.similarity_utils.is_subset(text, kept_doc.page_content) or 
                                     self.similarity_utils.is_subset(kept_doc.page_content, text))
                
                if similarity > self.similarity_threshold or is_subset_relation:
                    is_duplicate = True
                    similar_to = kept_doc
                    break
            
            if is_duplicate:
                removed_docs.append({
                    "doc": doc,
                    "reason": f"similarity: {similarity:.2f}" if 'similarity' in locals() else "subset relation",
                    "similar_to": similar_to
                })
            else:
                unique_docs.append(doc)
                seen_texts.add(text)
                
        return unique_docs, removed_docs
    
    def prioritize_by_comparator_coverage(self, docs, final_k=10):
        """
        Score and prioritize documents to maximize comparator coverage.
        """
        all_comparators = set()
        doc_comparators = []
        
        for doc in docs:
            comparators = self.similarity_utils.extract_potential_comparators(doc.page_content)
            all_comparators.update(comparators)
            doc_comparators.append((doc, comparators))
        
        selected_docs = []
        covered_comparators = set()
        skipped_docs = []
        
        while doc_comparators and len(selected_docs) < final_k:
            best_idx = -1
            best_unique_count = -1
            
            for idx, (_, comparators) in enumerate(doc_comparators):
                unique_count = len(comparators - covered_comparators)
                if unique_count > best_unique_count:
                    best_unique_count = unique_count
                    best_idx = idx
            
            if best_idx >= 0:
                doc, comparators = doc_comparators.pop(best_idx)
                selected_docs.append(doc)
                covered_comparators.update(comparators)
            else:
                if doc_comparators:
                    doc, comparators = doc_comparators.pop(0)
                    selected_docs.append(doc)
        
        for doc, comparators in doc_comparators:
            skipped_docs.append((doc, comparators))
            
        return selected_docs, skipped_docs, covered_comparators


class ChunkRetriever:
    """
    Retriever with specialized methods for HTA submissions vs clinical guidelines.
    Enhanced to support source type filtering with distinct retrieval strategies.
    Now supports split retrieval for Population & Comparator vs Outcomes.
    """
    def __init__(self, vectorstore, results_output_dir="results"):
        self.vectorstore = vectorstore
        self.chroma_collection = self.vectorstore._collection
        self.deduplicator = DocumentDeduplicator()
        self.similarity_utils = TextSimilarityUtils()
        self.context_manager = ContextManager()
        self.results_output_dir = results_output_dir
        self.chunks_output_dir = os.path.join(results_output_dir, "chunks")
        os.makedirs(self.chunks_output_dir, exist_ok=True)

    def _build_filter(self, country: str, source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Build proper ChromaDB filter combining country and optional source type.
        """
        if source_type:
            return {"$and": [{"country": country}, {"source_type": source_type}]}
        return {"country": country}

    def primary_filter_by_country(
        self, 
        country: str,
        source_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Filter by country with optional source type filter.
        """
        filter_dict = self._build_filter(country, source_type)
        try:
            result = self.chroma_collection.get(where=filter_dict, limit=limit)
            return [
                {"text": txt, "metadata": meta}
                for txt, meta in zip(result.get("documents", []), result.get("metadatas", []))
            ]
        except Exception as e:
            print(f"Error in primary_filter_by_country for {country} with source_type {source_type}: {e}")
            return []

    def hta_population_comparator_retrieval(
        self,
        query: str,
        country: str,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 50,
        final_k: int = 20,
        population_boost: float = 6.0,
        comparator_boost: float = 6.0,
        biomarker_boost: float = 5.0,
        mutation_boost_terms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for HTA submissions focusing on Population and Comparator elements.
        """
        filter_dict = self._build_filter(country, "hta_submission")
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k,
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in HTA population/comparator retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        unique_docs, _ = self.deduplicator.deduplicate_documents(docs)
        if not unique_docs:
            return []

        population_keywords = set([
            'population', 'patients', 'eligibility', 'inclusion', 'exclusion',
            'biomarker', 'mutation', 'testing', 'line of therapy', 'prior treatment',
            'disease stage', 'histology', 'performance status', 'sub-population'
        ])
        
        comparator_keywords = set([
            'comparator', 'comparison', 'versus', 'compared to', 'alternative',
            'standard of care', 'best supportive care', 'placebo', 'control arm',
            'appropriate comparator therapy', 'treatment comparison', 'indirect comparison'
        ])
        
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}

        scored_docs = []
        for i, doc in enumerate(unique_docs):
            score = (len(unique_docs) - i)
            
            heading_lower = (doc.metadata.get("heading") or "").lower()
            text_lower = (doc.page_content or "").lower()
            
            population_matches = sum(1 for keyword in population_keywords if keyword in text_lower)
            if population_matches > 0:
                score += population_boost * min(population_matches, 3)
            
            comparator_matches = sum(1 for keyword in comparator_keywords if keyword in text_lower)
            if comparator_matches > 0:
                score += comparator_boost * min(comparator_matches, 3)
            
            biomarker_terms = ['biomarker', 'mutation', 'testing', 'molecular', 'genetic']
            biomarker_matches = sum(1 for term in biomarker_terms if term in text_lower)
            if biomarker_matches > 0:
                score += biomarker_boost * min(biomarker_matches, 2)
            
            if any(k in heading_lower for k in heading_set):
                score += 4.0
            
            if any(d in text_lower for d in drug_set):
                score += 8.0
            
            if any(m in text_lower for m in mutation_set):
                score += 7.0
            
            if any(term in heading_lower for term in ['population', 'comparator', 'comparison', 'versus']):
                score += 8.0
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:final_k * 2]]

        selected_docs, _, _ = self.deduplicator.prioritize_by_comparator_coverage(
            top_docs, final_k=final_k
        )

        return [self._format_chunk_with_metadata(doc) for doc in selected_docs]

    def hta_outcomes_retrieval(
        self,
        query: str,
        country: str,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 40,
        final_k: int = 15,
        efficacy_boost: float = 5.0,
        safety_boost: float = 4.0,
        endpoint_boost: float = 6.0,
        mutation_boost_terms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for HTA submissions focusing on Outcomes elements.
        """
        filter_dict = self._build_filter(country, "hta_submission")
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k,
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in HTA outcomes retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        unique_docs, _ = self.deduplicator.deduplicate_documents(docs)
        if not unique_docs:
            return []

        efficacy_keywords = set([
            'efficacy', 'overall survival', 'progression-free survival', 'response rate',
            'duration of response', 'time to progression', 'disease control rate',
            'objective response', 'complete response', 'partial response',
            'quality of life', 'qol', 'patient reported outcomes', 'functional status',
            'health-related quality of life', 'hrqol', 'patient reported outcome measures',
            'proms', 'functional assessment', 'symptom burden'
        ])
        
        safety_keywords = set([
            'safety', 'adverse events', 'toxicity', 'tolerability', 'side effects',
            'serious adverse events', 'treatment-emergent', 'grade 3', 'grade 4'
        ])
        
        endpoint_keywords = set([
            'primary endpoint', 'secondary endpoint', 'outcomes', 'endpoints',
            'statistical analysis', 'hazard ratio', 'confidence interval', 'p-value'
        ])
        
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}

        scored_docs = []
        for i, doc in enumerate(unique_docs):
            score = (len(unique_docs) - i)
            
            heading_lower = (doc.metadata.get("heading") or "").lower()
            text_lower = (doc.page_content or "").lower()
            
            efficacy_matches = sum(1 for keyword in efficacy_keywords if keyword in text_lower)
            if efficacy_matches > 0:
                score += efficacy_boost * min(efficacy_matches, 4)
            
            safety_matches = sum(1 for keyword in safety_keywords if keyword in text_lower)
            if safety_matches > 0:
                score += safety_boost * min(safety_matches, 3)
            
            endpoint_matches = sum(1 for keyword in endpoint_keywords if keyword in text_lower)
            if endpoint_matches > 0:
                score += endpoint_boost * min(endpoint_matches, 3)
            
            if any(k in heading_lower for k in heading_set):
                score += 3.0
            
            if any(m in text_lower for m in mutation_set):
                score += 5.0
            
            if any(term in heading_lower for term in ['outcomes', 'efficacy', 'safety', 'endpoints']):
                score += 7.0
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [doc for doc, _ in scored_docs[:final_k]]

        return [self._format_chunk_with_metadata(doc) for doc in selected_docs]

    def clinical_population_comparator_retrieval(
        self,
        query: str,
        country: str,
        initial_k: int = 70,
        final_k: int = 18,
        strict_filtering: bool = True,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        heading_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for clinical guidelines focusing on Population and Comparator elements.
        """
        filter_dict = self._build_filter(country, "clinical_guideline")
        
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k * 2,
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in clinical guideline population/comparator retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        filtered_docs = []
        if strict_filtering and required_terms:
            for doc in docs:
                text_lower = doc.page_content.lower()
                heading_lower = (doc.metadata.get("heading") or "").lower()
                combined_text = text_lower + " " + heading_lower
                
                has_any_required = False
                for term_group in required_terms:
                    group_matched = False
                    for pattern in term_group:
                        if re.search(pattern, combined_text, re.IGNORECASE):
                            group_matched = True
                            break
                    if group_matched:
                        has_any_required = True
                        break
                
                if has_any_required:
                    filtered_docs.append(doc)
        else:
            filtered_docs = docs

        if not filtered_docs:
            print(f"No clinical guideline population/comparator chunks found for {country} after filtering")
            return []

        unique_docs, _ = self.deduplicator.deduplicate_documents(filtered_docs)
        if not unique_docs:
            return []

        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}

        scored_docs = []
        for i, doc in enumerate(unique_docs):
            score = (len(unique_docs) - i)
            
            text_lower = doc.page_content.lower()
            heading_lower = (doc.metadata.get("heading") or "").lower()
            
            mutation_matches = sum(1 for m in mutation_set if m in text_lower)
            if mutation_matches > 0:
                score += 10.0 * mutation_matches
            
            if any(m in heading_lower for m in mutation_set):
                score += 12.0
            
            population_terms = ['patients', 'population', 'eligible', 'biomarker', 'mutation', 'line of therapy']
            population_boost = sum(3.0 for term in population_terms if term in text_lower)
            score += population_boost
            
            comparator_terms = ['alternative', 'versus', 'compared', 'standard', 'option']
            comparator_boost = sum(4.0 for term in comparator_terms if term in text_lower)
            score += comparator_boost
            
            recommendation_terms = ['recommend', 'should', 'guideline', 'treatment', 'therapy']
            recommendation_boost = sum(2.5 for term in recommendation_terms if term in text_lower)
            score += recommendation_boost
            
            if any(k in heading_lower for k in heading_set):
                score += 5.0
            
            if any(d in text_lower for d in drug_set):
                score += 6.0
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored_docs[:final_k]]

        return [self._format_chunk_with_metadata(doc) for doc in final_docs]

    def clinical_outcomes_retrieval(
        self,
        query: str,
        country: str,
        initial_k: int = 60,
        final_k: int = 12,
        strict_filtering: bool = True,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        heading_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for clinical guidelines focusing on Outcomes elements.
        """
        filter_dict = self._build_filter(country, "clinical_guideline")
        
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k * 2,
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in clinical guideline outcomes retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        filtered_docs = []
        if strict_filtering and required_terms:
            for doc in docs:
                text_lower = doc.page_content.lower()
                heading_lower = (doc.metadata.get("heading") or "").lower()
                combined_text = text_lower + " " + heading_lower
                
                has_any_required = False
                for term_group in required_terms:
                    group_matched = False
                    for pattern in term_group:
                        if re.search(pattern, combined_text, re.IGNORECASE):
                            group_matched = True
                            break
                    if group_matched:
                        has_any_required = True
                        break
                
                if has_any_required:
                    filtered_docs.append(doc)
        else:
            filtered_docs = docs

        if not filtered_docs:
            print(f"No clinical guideline outcomes chunks found for {country} after filtering")
            return []

        unique_docs, _ = self.deduplicator.deduplicate_documents(filtered_docs)
        if not unique_docs:
            return []

        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}

        scored_docs = []
        for i, doc in enumerate(unique_docs):
            score = (len(unique_docs) - i)
            
            text_lower = doc.page_content.lower()
            heading_lower = (doc.metadata.get("heading") or "").lower()
            
            outcomes_terms = ['efficacy', 'response', 'survival', 'outcomes', 'benefit', 'improvement',
                            'quality of life', 'qol', 'patient reported outcomes', 'functional status',
                            'health-related quality of life', 'hrqol', 'patient reported outcome measures',
                            'proms', 'functional assessment', 'symptom burden']
            outcomes_boost = sum(4.0 for term in outcomes_terms if term in text_lower)
            score += outcomes_boost
            
            safety_terms = ['safety', 'adverse', 'toxicity', 'tolerability', 'side effects']
            safety_boost = sum(3.0 for term in safety_terms if term in text_lower)
            score += safety_boost
            
            evidence_terms = ['evidence', 'level', 'grade', 'recommendation', 'strength']
            evidence_boost = sum(3.0 for term in evidence_terms if term in text_lower)
            score += evidence_boost
            
            mutation_matches = sum(1 for m in mutation_set if m in text_lower)
            if mutation_matches > 0:
                score += 6.0 * mutation_matches
            
            if any(k in heading_lower for k in heading_set):
                score += 4.0
            
            if any(term in heading_lower for term in ['outcomes', 'efficacy', 'safety', 'response']):
                score += 8.0
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored_docs[:final_k]]

        return [self._format_chunk_with_metadata(doc) for doc in final_docs]

    def _format_chunk_with_metadata(self, doc) -> Dict[str, Any]:
        """
        Format a document chunk with comprehensive metadata for storage.
        """
        return {
            "text": doc.page_content,
            "metadata": {
                "heading": doc.metadata.get("heading", ""),
                "doc_id": doc.metadata.get("doc_id", ""),
                "country": doc.metadata.get("country", ""),
                "source_type": doc.metadata.get("source_type", ""),
                "start_page": doc.metadata.get("start_page"),
                "end_page": doc.metadata.get("end_page"),
                "created_date": doc.metadata.get("created_date", ""),
                "folder_path": doc.metadata.get("folder_path", ""),
                "split_index": doc.metadata.get("split_index"),
                "text_length": len(doc.page_content),
                "potential_comparators": list(self.similarity_utils.extract_potential_comparators(doc.page_content))
            }
        }

    def retrieve_population_comparator_chunks(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        initial_k: int = 50,
        final_k: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks focused on Population and Comparator elements by country.
        """
        results_by_country: Dict[str, List[Dict[str, Any]]] = {}
        for country in countries:
            if source_type == "hta_submission":
                chunks = self.hta_population_comparator_retrieval(
                    query=query,
                    country=country,
                    heading_keywords=heading_keywords,
                    drug_keywords=drug_keywords,
                    initial_k=initial_k,
                    final_k=final_k,
                    mutation_boost_terms=mutation_boost_terms
                )
            elif source_type == "clinical_guideline":
                chunks = self.clinical_population_comparator_retrieval(
                    query=query,
                    country=country,
                    initial_k=initial_k,
                    final_k=final_k,
                    required_terms=required_terms,
                    mutation_boost_terms=mutation_boost_terms,
                    drug_keywords=drug_keywords,
                    heading_keywords=heading_keywords
                )
            else:
                chunks = []
            
            results_by_country[country] = chunks
        return results_by_country

    def retrieve_outcomes_chunks(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        initial_k: int = 40,
        final_k: int = 15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks focused on Outcomes elements by country.
        """
        results_by_country: Dict[str, List[Dict[str, Any]]] = {}
        for country in countries:
            if source_type == "hta_submission":
                chunks = self.hta_outcomes_retrieval(
                    query=query,
                    country=country,
                    heading_keywords=heading_keywords,
                    drug_keywords=drug_keywords,
                    initial_k=initial_k,
                    final_k=final_k,
                    mutation_boost_terms=mutation_boost_terms
                )
            elif source_type == "clinical_guideline":
                chunks = self.clinical_outcomes_retrieval(
                    query=query,
                    country=country,
                    initial_k=initial_k,
                    final_k=final_k,
                    required_terms=required_terms,
                    mutation_boost_terms=mutation_boost_terms,
                    drug_keywords=drug_keywords,
                    heading_keywords=heading_keywords
                )
            else:
                chunks = []
            
            results_by_country[country] = chunks
        return results_by_country

    def save_retrieval_results(
        self,
        results_by_country: Dict[str, List[Dict[str, Any]]],
        source_type: str,
        retrieval_type: str = "combined",
        query: str = "",
        timestamp: Optional[str] = None,
        indication: Optional[str] = None
    ):
        """
        Save retrieval results to JSON file organized by source type, retrieval type and country.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        output_data = {
            "retrieval_metadata": {
                "timestamp": timestamp,
                "source_type": source_type,
                "retrieval_type": retrieval_type,
                "query": query,
                "indication": indication,
                "total_countries": len(results_by_country),
                "total_chunks": sum(len(chunks) for chunks in results_by_country.values())
            },
            "results_by_country": {}
        }
        
        for country, chunks in results_by_country.items():
            output_data["results_by_country"][country] = {
                "country_metadata": {
                    "country_code": country,
                    "chunk_count": len(chunks),
                    "total_text_length": sum(len(chunk["text"]) for chunk in chunks),
                    "unique_documents": len(set(chunk["metadata"]["doc_id"] for chunk in chunks)),
                    "unique_headings": len(set(chunk["metadata"]["heading"] for chunk in chunks if chunk["metadata"]["heading"]))
                },
                "chunks": chunks
            }
        
        if indication:
            indication_short = indication.split()[0].lower() if indication else "unknown"
            filename = f"{source_type}_{retrieval_type}_{indication_short}_retrieval_results.json"
        else:
            filename = f"{source_type}_{retrieval_type}_retrieval_results.json"
        filepath = os.path.join(self.chunks_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {retrieval_type} retrieval results to {filepath}")
        return filepath

    def diagnose_vectorstore(self, limit: int = 100):
        """
        Diagnose the vectorstore by checking available metadata fields and values.
        """
        try:
            result = self.chroma_collection.get(limit=limit, include=['metadatas'])
            metadatas = result.get('metadatas', []) or []
            print(f"=== VECTORSTORE DIAGNOSTICS ===")
            print(f"Total documents sampled: {len(metadatas)}")
            all_fields = set()
            country_values = set()
            source_type_values = set()
            for metadata in metadatas:
                if metadata:
                    all_fields.update(metadata.keys())
                    if 'country' in metadata:
                        country_values.add(metadata['country'])
                    if 'source_type' in metadata:
                        source_type_values.add(metadata['source_type'])
            print(f"Available metadata fields: {sorted(all_fields)}")
            print(f"Countries found: {sorted(country_values)}")
            print(f"Source types found: {sorted(source_type_values)}")
            print(f"\nSample metadata entries:")
            for i, metadata in enumerate(metadatas[:5]):
                print(f"  {i+1}: {metadata}")
            return {
                'total_docs': len(metadatas),
                'metadata_fields': sorted(all_fields),
                'countries': sorted(country_values),
                'source_types': sorted(source_type_values)
            }
        except Exception as e:
            print(f"Error diagnosing vectorstore: {e}")
            return None

    def test_simple_retrieval(self, country: str = "EN", limit: int = 5):
        """
        Test simple retrieval without any filters to see if basic functionality works.
        """
        try:
            print(f"=== TESTING SIMPLE RETRIEVAL ===")
            print(f"Testing basic retrieval for country: {country}")
            result_no_filter = self.chroma_collection.get(limit=limit)
            print(f"Documents without filter: {len(result_no_filter.get('documents', []))}")
            result_country = self.chroma_collection.get(where={"country": country}, limit=limit)
            print(f"Documents with country={country}: {len(result_country.get('documents', []))}")
            docs_no_filter = self.vectorstore.similarity_search(query="treatment", k=limit)
            print(f"Vector similarity without filter: {len(docs_no_filter)} docs")
            docs_with_filter = self.vectorstore.similarity_search(query="treatment", k=limit, filter={"country": country})
            print(f"Vector similarity with country filter: {len(docs_with_filter)} docs")
            return True
        except Exception as e:
            print(f"Error in simple retrieval test: {e}")
            return False


class ContextManager:
    """
    Class to handle adaptive context management for LLM prompts.
    """
    def __init__(self, max_tokens=12000):
        self.max_tokens = max_tokens
        self.similarity_utils = TextSimilarityUtils()
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")
    
    def count_tokens(self, text):
        """Count tokens in text using the current encoding."""
        return len(self.encoding.encode(text))
    
    def process_chunks(self, chunks):
        """Process chunks to estimate tokens and extract comparators."""
        processed = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            token_count = self.count_tokens(text)
            comparators = self.similarity_utils.extract_potential_comparators(text)
            
            processed.append({
                "text": text,
                "tokens": token_count,
                "comparators": comparators,
                "metadata": chunk.get("metadata", {})
            })
        
        return processed
    
    def build_optimal_context(self, processed_chunks):
        """
        Build optimal context block maximizing comparator coverage
        while respecting token limits.
        """
        context_parts = []
        current_tokens = 0
        covered_comparators = set()
        
        all_comparators = set()
        for chunk in processed_chunks:
            all_comparators.update(chunk["comparators"])
        
        def sort_key(chunk):
            unique_count = len(chunk["comparators"] - covered_comparators)
            return unique_count
        
        remaining_chunks = list(processed_chunks)
        while remaining_chunks and current_tokens < self.max_tokens:
            remaining_chunks.sort(key=sort_key, reverse=True)
            chunk = remaining_chunks.pop(0)
            
            if current_tokens + chunk["tokens"] > self.max_tokens:
                new_comparators = chunk["comparators"] - covered_comparators
                if not new_comparators or current_tokens + chunk["tokens"] > self.max_tokens * 1.1:
                    continue
            
            context_parts.append(chunk["text"])
            current_tokens += chunk["tokens"]
            covered_comparators.update(chunk["comparators"])
            
            if covered_comparators >= all_comparators:
                break
        
        return "\n\n".join(context_parts)