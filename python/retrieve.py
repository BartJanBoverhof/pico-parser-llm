import os
import json
import re
import tiktoken
from typing import List, Dict, Any, Optional, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
import pandas as pd
from datetime import datetime


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
        # Clean and tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # If most of text1's tokens are in text2, consider it a subset
        overlap_ratio = len(tokens1.intersection(tokens2)) / len(tokens1) if tokens1 else 0
        return overlap_ratio > 0.9  # 90% of tokens are contained

    @staticmethod
    def extract_potential_comparators(text):
        """
        Extract potential drug names/comparators from text using pattern matching.
        """
        words = text.split()
        capitalized_words = []
        
        # Find capitalized words that might be drug names
        for i, word in enumerate(words):
            # Check if this is a potential sentence start or after a space
            if (i > 0 and words[i-1][-1] in '.!?') or i == 0:
                # Clean the word of punctuation
                clean_word = word.strip('.,;:()[]{}')
                if clean_word and clean_word[0].isupper() and len(clean_word) > 1 and clean_word.lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'by', 'to', 'of']:
                    capitalized_words.append(clean_word)
        
        # Find words followed by dosages (simple pattern)
        dosage_pattern = r'\b\w+\s+\d+\s*(?:mg|mcg|g|ml)\b'
        dosages = re.findall(dosage_pattern, text)
        
        # Find drug name suffixes
        suffix_pattern = r'\b\w+(?:mab|nib|zumab|tinib|ciclib|parib|vastatin)\b'
        suffix_matches = re.findall(suffix_pattern, text.lower())
        
        # Combine all matches
        all_matches = capitalized_words + dosages + suffix_matches
        
        # Filter out common words that aren't likely drug names
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
            # Simple deduplication for identical content
            text = doc.page_content.strip()
            if text in seen_texts:
                removed_docs.append({
                    "doc": doc,
                    "reason": "exact duplicate",
                    "similar_to": None
                })
                continue
                
            # Check for near-duplicates or subset relationships
            is_duplicate = False
            similar_to = None
            
            for kept_doc in unique_docs:
                # Skip comparison if preserving country diversity and documents are from different countries
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
        # Extract all potential comparators from documents
        all_comparators = set()
        doc_comparators = []
        
        for doc in docs:
            comparators = self.similarity_utils.extract_potential_comparators(doc.page_content)
            all_comparators.update(comparators)
            doc_comparators.append((doc, comparators))
        
        # Prioritize documents with unique comparators
        selected_docs = []
        covered_comparators = set()
        skipped_docs = []
        
        # Sort by number of unique comparators (most unique first)
        while doc_comparators and len(selected_docs) < final_k:
            # Find document with most uncovered comparators
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
                # If no more unique comparators, just take the next document
                if doc_comparators:
                    doc, comparators = doc_comparators.pop(0)
                    selected_docs.append(doc)
        
        # Remaining docs weren't selected
        for doc, comparators in doc_comparators:
            skipped_docs.append((doc, comparators))
            
        return selected_docs, skipped_docs, covered_comparators


class ChunkRetriever:
    """
    Retriever with specialized methods for HTA submissions vs clinical guidelines.
    Enhanced to support source type filtering with distinct retrieval strategies.
    """
    def __init__(self, vectorstore, results_output_dir="results"):
        self.vectorstore = vectorstore
        self.chroma_collection = self.vectorstore._collection
        self.deduplicator = DocumentDeduplicator()
        self.similarity_utils = TextSimilarityUtils()
        self.context_manager = ContextManager()
        self.results_output_dir = results_output_dir
        os.makedirs(self.results_output_dir, exist_ok=True)

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

    def hta_submission_retrieval(
        self,
        query: str,
        country: str,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10,
        comparator_boost: float = 5.0,
        pico_boost: float = 4.0,
        drug_boost: float = 8.0,
        mutation_boost_terms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for HTA submissions leveraging their structured nature.
        Focuses on PICO elements, comparators, and treatment information.
        """
        filter_dict = self._build_filter(country, "hta_submission")
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k,
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in HTA submission retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        # Deduplicate similar chunks
        unique_docs, _ = self.deduplicator.deduplicate_documents(docs)
        if not unique_docs:
            return []

        # HTA-specific scoring with emphasis on structured elements
        hta_structure_keywords = set([
            'comparator', 'comparison', 'versus', 'compared to', 'alternative',
            'population', 'intervention', 'outcome', 'endpoint', 'efficacy',
            'safety', 'pico', 'treatment', 'therapy', 'medicinal product',
            'appropriate comparator therapy', 'designation of therapy'
        ])
        
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}

        scored_docs = []
        for i, doc in enumerate(unique_docs):
            # Base similarity score
            score = (len(unique_docs) - i)
            
            heading_lower = (doc.metadata.get("heading") or "").lower()
            text_lower = (doc.page_content or "").lower()
            
            # HTA structure boost - prioritize PICO and comparator sections
            structure_matches = sum(1 for keyword in hta_structure_keywords if keyword in heading_lower)
            if structure_matches > 0:
                score += comparator_boost * structure_matches
            
            # PICO element boost in content
            pico_content_matches = sum(1 for keyword in hta_structure_keywords if keyword in text_lower)
            if pico_content_matches > 0:
                score += pico_boost * min(pico_content_matches, 3)  # Cap to avoid over-boosting
            
            # Heading keyword boost
            if any(k in heading_lower for k in heading_set):
                score += 3.0
            
            # Drug keyword boost
            if any(d in text_lower for d in drug_set):
                score += drug_boost
            
            # Mutation-specific boost
            if any(m in text_lower for m in mutation_set):
                score += 6.0
            
            # Additional boost for sections explicitly about comparators
            if any(term in heading_lower for term in ['comparator', 'comparison', 'versus', 'alternative']):
                score += 6.0
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:final_k * 2]]

        # Prioritize by comparator coverage for HTA submissions
        selected_docs, _, _ = self.deduplicator.prioritize_by_comparator_coverage(
            top_docs, final_k=final_k
        )

        return [self._format_chunk_with_metadata(doc) for doc in selected_docs]

    def clinical_guideline_retrieval(
        self,
        query: str,
        country: str,
        initial_k: int = 50,
        final_k: int = 10,
        strict_filtering: bool = True,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        heading_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Specialized retrieval for clinical guidelines with configurable filtering.
        Enhanced with proper required_terms filtering for mutation-specific retrieval.
        """
        filter_dict = self._build_filter(country, "clinical_guideline")
        
        # Use broader initial retrieval for guidelines since relevant content is sparse
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=initial_k * 2,  # Cast wider net for guidelines
                filter=filter_dict,
            )
        except Exception as e:
            print(f"Error in clinical guideline retrieval for {country}: {e}")
            return []
        
        if not docs:
            return []

        # Apply required terms filtering if specified
        filtered_docs = []
        if strict_filtering and required_terms:
            for doc in docs:
                text_lower = doc.page_content.lower()
                heading_lower = (doc.metadata.get("heading") or "").lower()
                combined_text = text_lower + " " + heading_lower
                
                # Check if all required term groups are satisfied
                # Each group is an OR condition, all groups must be satisfied (AND)
                has_all_required = True
                for term_group in required_terms:
                    # At least one pattern from the group must match
                    group_matched = False
                    for pattern in term_group:
                        if re.search(pattern, combined_text, re.IGNORECASE):
                            group_matched = True
                            break
                    if not group_matched:
                        has_all_required = False
                        break
                
                if has_all_required:
                    filtered_docs.append(doc)
        else:
            filtered_docs = docs

        if not filtered_docs:
            print(f"No clinical guideline chunks found for {country} after required terms filtering")
            return []

        # Deduplicate after filtering
        unique_docs, _ = self.deduplicator.deduplicate_documents(filtered_docs)
        if not unique_docs:
            return []

        # Prepare boost term sets
        mutation_set = set((mutation_boost_terms or []))
        mutation_set = {m.lower() for m in mutation_set}
        drug_set = set((drug_keywords or []))
        drug_set = {d.lower() for d in drug_set}
        heading_set = set((heading_keywords or []))
        heading_set = {k.lower() for k in heading_set}

        # Clinical guideline specific scoring
        scored_docs = []
        for i, doc in enumerate(unique_docs):
            score = (len(unique_docs) - i)
            
            text_lower = doc.page_content.lower()
            heading_lower = (doc.metadata.get("heading") or "").lower()
            
            # Strong boost for mutation-specific content
            mutation_matches = sum(1 for m in mutation_set if m in text_lower)
            if mutation_matches > 0:
                score += 8.0 * mutation_matches
            
            # Boost for mutation in heading
            if any(m in heading_lower for m in mutation_set):
                score += 10.0
            
            # Boost for treatment recommendations
            recommendation_terms = ['recommend', 'should', 'guideline', 'treatment', 'therapy', 'algorithm', 'indication', 'eligible']
            recommendation_boost = sum(2.0 for term in recommendation_terms if term in text_lower)
            score += recommendation_boost
            
            # Boost for progression/line therapy content
            progression_terms = ['second-line', 'second line', 'progression', 'refractory', 'resistant', 'subsequent', 'previously treated', 'after']
            progression_boost = sum(3.0 for term in progression_terms if term in text_lower)
            score += progression_boost
            
            # Boost for heading keywords
            if any(k in heading_lower for k in heading_set):
                score += 4.0
            
            # Boost for drug keywords
            if any(d in text_lower for d in drug_set):
                score += 5.0
            
            # Penalize overly generic content if mutation terms are required
            if required_terms and mutation_set:
                # Check if this is generic lung cancer content without mutation specifics
                generic_indicators = ['general lung cancer', 'all patients', 'tumor board', 'multidisciplinary']
                has_generic = any(indicator in text_lower for indicator in generic_indicators)
                has_mutation = any(m in text_lower for m in mutation_set)
                if has_generic and not has_mutation:
                    score -= 10.0  # Penalize generic content
            
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top scored documents up to final_k
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

    def vector_similarity_search(
        self,
        query: str,
        country: str,
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10,
        heading_boost: float = 3.0,
        drug_boost: float = 8.0
    ) -> List[Dict[str, Any]]:
        """
        General vector similarity search with keyword boosts.
        Routes to specialized methods based on source_type.
        """
        if source_type == "hta_submission":
            return self.hta_submission_retrieval(
                query=query,
                country=country,
                heading_keywords=heading_keywords,
                drug_keywords=drug_keywords,
                initial_k=initial_k,
                final_k=final_k,
                mutation_boost_terms=mutation_boost_terms
            )
        elif source_type == "clinical_guideline":
            return self.clinical_guideline_retrieval(
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
            # Fallback to general retrieval
            filter_dict = self._build_filter(country, source_type)
            try:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=initial_k,
                    filter=filter_dict,
                )
            except Exception as e:
                print(f"Error in vector similarity search for {country}: {e}")
                return []
            
            if not docs:
                return []

            # Deduplicate similar chunks
            unique_docs, _ = self.deduplicator.deduplicate_documents(docs)
            if not unique_docs:
                return []

            # Score chunks by heading and drug keyword relevance
            keyword_set = set((heading_keywords or []))
            keyword_set = {k.lower() for k in keyword_set}
            drug_set = set((drug_keywords or []))
            drug_set = {d.lower() for d in drug_set}

            scored_docs = []
            for i, doc in enumerate(unique_docs):
                score = (len(unique_docs) - i)  # slight bias for earlier items
                heading_lower = (doc.metadata.get("heading") or "").lower()
                if any(k in heading_lower for k in keyword_set):
                    score += heading_boost
                text_lower = (doc.page_content or "").lower()
                if any(d in text_lower for d in drug_set):
                    score += drug_boost
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, _ in scored_docs[:final_k * 2]]

            # Prioritize by comparator coverage to maximize variety of comparators/drugs
            selected_docs, _, _ = self.deduplicator.prioritize_by_comparator_coverage(
                top_docs, final_k=final_k
            )

            return [self._format_chunk_with_metadata(doc) for doc in selected_docs]

    def retrieve_pico_chunks(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks by country using specialized retrieval based on source type.
        """
        results_by_country: Dict[str, List[Dict[str, Any]]] = {}
        for country in countries:
            chunks = self.vector_similarity_search(
                query=query,
                country=country,
                source_type=source_type,
                heading_keywords=heading_keywords,
                drug_keywords=drug_keywords,
                required_terms=required_terms,
                mutation_boost_terms=mutation_boost_terms,
                initial_k=initial_k,
                final_k=final_k
            )
            results_by_country[country] = chunks
        return results_by_country

    def save_retrieval_results(
        self,
        results_by_country: Dict[str, List[Dict[str, Any]]],
        source_type: str,
        query: str,
        timestamp: Optional[str] = None
    ):
        """
        Save retrieval results to JSON file organized by source type and country.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Prepare structured output
        output_data = {
            "retrieval_metadata": {
                "timestamp": timestamp,
                "source_type": source_type,
                "query": query,
                "total_countries": len(results_by_country),
                "total_chunks": sum(len(chunks) for chunks in results_by_country.values())
            },
            "results_by_country": {}
        }
        
        # Organize results by country
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
        
        # Save to file
        filename = f"{source_type}_retrieval_results.json"
        filepath = os.path.join(self.results_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved retrieval results to {filepath}")
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

    def test_retrieval(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10
    ) -> Dict[str, Any]:
        """
        Test the retrieval pipeline with specialized methods and save results to JSON.
        """
        chunks_by_country = self.retrieve_pico_chunks(
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

        # Save results to JSON
        timestamp = datetime.now().isoformat()
        self.save_retrieval_results(
            results_by_country=chunks_by_country,
            source_type=source_type or "general",
            query=query,
            timestamp=timestamp
        )

        # Print simple summary
        print(f"\n=== RETRIEVAL RESULTS ===")
        print(f"Query: {query}")
        print(f"Source type: {source_type or 'All sources'}")

        summary: Dict[str, int] = {}
        total_chunks = 0
        for country, chunks in chunks_by_country.items():
            count = len(chunks)
            summary[country] = count
            total_chunks += count
            print(f"{country}: {count} chunks")

        print(f"Total: {total_chunks} chunks")
        print("=" * 25)

        return {
            "summary": summary,
            "chunks_by_country": chunks_by_country,
            "timestamp": timestamp
        }


class ContextManager:
    """
    Class to handle adaptive context management for LLM prompts.
    """
    def __init__(self, max_tokens=12000):
        self.max_tokens = max_tokens
        self.similarity_utils = TextSimilarityUtils()
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")  # Fallback
    
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
        
        # Get all potential comparators
        all_comparators = set()
        for chunk in processed_chunks:
            all_comparators.update(chunk["comparators"])
        
        # Sort by unique comparator coverage
        def sort_key(chunk):
            unique_count = len(chunk["comparators"] - covered_comparators)
            return unique_count
        
        # First pass: include chunks with unique comparators
        remaining_chunks = list(processed_chunks)
        while remaining_chunks and current_tokens < self.max_tokens:
            # Resort each time as covered_comparators changes
            remaining_chunks.sort(key=sort_key, reverse=True)
            chunk = remaining_chunks.pop(0)
            
            # Skip if adding would exceed token limit
            if current_tokens + chunk["tokens"] > self.max_tokens:
                new_comparators = chunk["comparators"] - covered_comparators
                # Only include if it has unique comparators and we're not too far over limit
                if not new_comparators or current_tokens + chunk["tokens"] > self.max_tokens * 1.1:
                    continue
            
            context_parts.append(chunk["text"])
            current_tokens += chunk["tokens"]
            covered_comparators.update(chunk["comparators"])
            
            # If we've covered all comparators, we can stop
            if covered_comparators >= all_comparators:
                break
        
        return "\n\n".join(context_parts)


class PICOExtractor:
    """
    PICOExtractor with improved chunk deduplication and adaptive context handling.
    Enhanced with JSON storage for organized PICO results and indication parameterization.
    """
    def __init__(
        self,
        chunk_retriever,
        system_prompt: str,
        user_prompt_template: str,
        model_name: str = "gpt-4o-mini",
        results_output_dir: str = "results"
    ):
        self.chunk_retriever = chunk_retriever
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.context_manager = ContextManager()
        self.results_output_dir = results_output_dir
        os.makedirs(self.results_output_dir, exist_ok=True)

    def extract_picos(
        self,
        countries: List[str],
        query: str,
        source_type: Optional[str] = None,
        initial_k: int = 10,
        final_k: int = 5,
        heading_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs using improved context management and LLM with JSON storage.
        """
        results = []
        timestamp = datetime.now().isoformat()

        # Optionally override the model
        llm_to_use = self.llm
        if model_override:
            if isinstance(model_override, str):
                llm_to_use = ChatOpenAI(model_name=model_override, temperature=0)
            elif isinstance(model_override, ChatOpenAI):
                llm_to_use = model_override

        # Retrieve chunks for all countries
        results_dict = self.chunk_retriever.retrieve_pico_chunks(
            query=query,
            countries=countries,
            source_type=source_type,
            heading_keywords=heading_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k
        )

        # Save retrieval results
        self.chunk_retriever.save_retrieval_results(
            results_by_country=results_dict,
            source_type=source_type or "general",
            query=query,
            timestamp=timestamp
        )

        # Process each country
        for country in countries:
            country_chunks = results_dict.get(country, [])
            if not country_chunks:
                print(f"No chunks retrieved for {country} with source_type: {source_type}")
                continue

            # Process chunks with context manager
            processed_chunks = self.context_manager.process_chunks(country_chunks)
            context_block = self.context_manager.build_optimal_context(processed_chunks)

            # Prepare system and user messages
            system_msg = SystemMessage(content=self.system_prompt)
            user_msg_text = self.user_prompt_template.format(
                context_block=context_block,
                country=country
            )
            user_msg = HumanMessage(content=user_msg_text)

            # LLM call
            try:
                llm_response: BaseMessage = llm_to_use([system_msg, user_msg])
            except Exception as exc:
                print(f"LLM call failed for {country}: {exc}")
                continue

            answer_text = getattr(llm_response, 'content', str(llm_response))

            # Parse JSON response
            try:
                parsed_json = json.loads(answer_text)
            except json.JSONDecodeError:
                # Retry once with explicit instruction to fix JSON
                fix_msg = HumanMessage(content="Please correct and return valid JSON in the specified format only.")
                try:
                    fix_response = llm_to_use([system_msg, user_msg, fix_msg])
                    fix_text = getattr(fix_response, 'content', str(fix_response))
                    parsed_json = json.loads(fix_text)
                except Exception as parse_err:
                    print(f"Failed to parse JSON for {country}: {parse_err}")
                    continue

            # Process and store results
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country
                parsed_json["SourceType"] = source_type
                parsed_json["RetrievalTimestamp"] = timestamp
                parsed_json["ChunksUsed"] = len(country_chunks)
                results.append(parsed_json)
            else:
                # Handle non-dict response
                wrapped_json = {
                    "Country": country,
                    "SourceType": source_type,
                    "RetrievalTimestamp": timestamp,
                    "ChunksUsed": len(country_chunks),
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                results.append(wrapped_json)

        # Save organized PICO results to JSON
        if results:
            self._save_pico_results(results, source_type, timestamp)

        return results

    def extract_picos_with_indication(
        self,
        countries: List[str],
        indication: str,
        source_type: Optional[str] = None,
        initial_k: int = 10,
        final_k: int = 5,
        heading_keywords: Optional[List[str]] = None,
        required_terms: Optional[List[List[str]]] = None,
        mutation_boost_terms: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs with parameterized indication for query and prompting.
        """
        results = []
        timestamp = datetime.now().isoformat()

        # Optionally override the model
        llm_to_use = self.llm
        if model_override:
            if isinstance(model_override, str):
                llm_to_use = ChatOpenAI(model_name=model_override, temperature=0)
            elif isinstance(model_override, ChatOpenAI):
                llm_to_use = model_override

        # Build query based on source type template from config
        from python.config import SOURCE_TYPE_CONFIGS
        if source_type and source_type in SOURCE_TYPE_CONFIGS:
            query_template = SOURCE_TYPE_CONFIGS[source_type]["query_template"]
            query = query_template.format(indication=indication)
        else:
            query = f"Find relevant PICO information for: {indication}"

        # Retrieve chunks for all countries
        results_dict = self.chunk_retriever.retrieve_pico_chunks(
            query=query,
            countries=countries,
            source_type=source_type,
            heading_keywords=heading_keywords,
            required_terms=required_terms,
            mutation_boost_terms=mutation_boost_terms,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k
        )

        # Save retrieval results
        self.chunk_retriever.save_retrieval_results(
            results_by_country=results_dict,
            source_type=source_type or "general",
            query=query,
            timestamp=timestamp
        )

        # Process each country
        for country in countries:
            country_chunks = results_dict.get(country, [])
            if not country_chunks:
                print(f"No chunks retrieved for {country} with source_type: {source_type}")
                continue

            # Process chunks with context manager
            processed_chunks = self.context_manager.process_chunks(country_chunks)
            context_block = self.context_manager.build_optimal_context(processed_chunks)

            # Prepare system and user messages with indication parameterization
            system_msg = SystemMessage(content=self.system_prompt)
            user_msg_text = self.user_prompt_template.format(
                indication=indication,
                context_block=context_block,
                country=country
            )
            user_msg = HumanMessage(content=user_msg_text)

            # LLM call
            try:
                llm_response: BaseMessage = llm_to_use([system_msg, user_msg])
            except Exception as exc:
                print(f"LLM call failed for {country}: {exc}")
                continue

            answer_text = getattr(llm_response, 'content', str(llm_response))

            # Parse JSON response
            try:
                parsed_json = json.loads(answer_text)
            except json.JSONDecodeError:
                # Retry once with explicit instruction to fix JSON
                fix_msg = HumanMessage(content="Please correct and return valid JSON in the specified format only.")
                try:
                    fix_response = llm_to_use([system_msg, user_msg, fix_msg])
                    fix_text = getattr(fix_response, 'content', str(fix_response))
                    parsed_json = json.loads(fix_text)
                except Exception as parse_err:
                    print(f"Failed to parse JSON for {country}: {parse_err}")
                    continue

            # Process and store results
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country
                parsed_json["SourceType"] = source_type
                parsed_json["Indication"] = indication
                parsed_json["RetrievalTimestamp"] = timestamp
                parsed_json["ChunksUsed"] = len(country_chunks)
                results.append(parsed_json)
            else:
                # Handle non-dict response
                wrapped_json = {
                    "Country": country,
                    "SourceType": source_type,
                    "Indication": indication,
                    "RetrievalTimestamp": timestamp,
                    "ChunksUsed": len(country_chunks),
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                results.append(wrapped_json)

        # Save organized PICO results to JSON
        if results:
            self._save_pico_results_with_indication(results, source_type, indication, timestamp)

        return results

    def _save_pico_results(self, results: List[Dict[str, Any]], source_type: str, timestamp: str):
        """
        Save PICO extraction results organized by source type and country.
        """
        # Organize results by source type and country
        organized_results = {
            "extraction_metadata": {
                "timestamp": timestamp,
                "source_type": source_type or "general",
                "total_countries": len(results),
                "model_used": self.model_name
            },
            "picos_by_country": {}
        }
        
        for result in results:
            country = result.get("Country", "unknown")
            picos = result.get("PICOs", [])
            
            organized_results["picos_by_country"][country] = {
                "country_metadata": {
                    "country_code": country,
                    "total_picos": len(picos),
                    "chunks_used": result.get("ChunksUsed", 0),
                    "extraction_timestamp": result.get("RetrievalTimestamp", timestamp)
                },
                "extracted_picos": picos
            }
        
        # Save to source-specific file
        source_prefix = source_type.replace("_", "") if source_type else "general"
        filename = f"{source_prefix}_picos_organized.json"
        filepath = os.path.join(self.results_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved organized PICO results to {filepath}")
        return filepath

    def _save_pico_results_with_indication(self, results: List[Dict[str, Any]], source_type: str, indication: str, timestamp: str):
        """
        Save PICO extraction results organized by source type, indication, and country.
        """
        # Organize results by source type, indication, and country
        organized_results = {
            "extraction_metadata": {
                "timestamp": timestamp,
                "source_type": source_type or "general",
                "indication": indication,
                "total_countries": len(results),
                "model_used": self.model_name
            },
            "picos_by_country": {}
        }
        
        for result in results:
            country = result.get("Country", "unknown")
            picos = result.get("PICOs", [])
            
            organized_results["picos_by_country"][country] = {
                "country_metadata": {
                    "country_code": country,
                    "indication": indication,
                    "total_picos": len(picos),
                    "chunks_used": result.get("ChunksUsed", 0),
                    "extraction_timestamp": result.get("RetrievalTimestamp", timestamp)
                },
                "extracted_picos": picos
            }
        
        # Save to indication and source-specific file
        source_prefix = source_type.replace("_", "") if source_type else "general"
        indication_short = indication.split()[0].lower() if indication else "unknown"
        filename = f"{source_prefix}_{indication_short}_picos_organized.json"
        filepath = os.path.join(self.results_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved organized PICO results with indication to {filepath}")
        return filepath