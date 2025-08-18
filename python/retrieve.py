import os
import json
import re
import tiktoken
from typing import List, Dict, Any, Optional, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
import pandas as pd


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
    Retriever with improved deduplication and adaptive context handling.
    Enhanced to support source type filtering.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chroma_collection = self.vectorstore._collection
        self.deduplicator = DocumentDeduplicator()
        self.similarity_utils = TextSimilarityUtils()
        self.context_manager = ContextManager()

    def _build_filter(self, country: str, source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Build proper ChromaDB filter combining country and optional source type.
        """
        if source_type:
            # Use $and operator for combining multiple conditions
            return {
                "$and": [
                    {"country": country},
                    {"source_type": source_type}
                ]
            }
        else:
            # Single condition doesn't need operator
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
            result = self.chroma_collection.get(
                where=filter_dict,
                limit=limit
            )
            return [
                {"text": txt, "metadata": meta}
                for txt, meta in zip(result["documents"], result["metadatas"])
            ]
        except Exception as e:
            print(f"Error in primary_filter_by_country for {country} with source_type {source_type}: {e}")
            return []

    def vector_similarity_search(
        self,
        query: str,
        country: str,
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10,
        heading_boost: float = 3.0,
        drug_boost: float = 8.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by vector similarity with keyword boosts.
        """
        # Build filter
        filter_dict = self._build_filter(country, source_type)
        
        try:
            # Get initial chunks via vector similarity
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
        keyword_set = set(kw.lower() for kw in (heading_keywords or []))
        drug_set = set(dr.lower() for dr in (drug_keywords or []))
        
        scored_docs = []
        for i, doc in enumerate(unique_docs):
            # Base score: higher for earlier docs
            base_score = (len(unique_docs) - i)
            
            # Heading boost
            heading_lower = doc.metadata.get("heading", "").lower()
            if any(k in heading_lower for k in keyword_set):
                base_score += heading_boost
                
            # Drug name boost
            text_lower = doc.page_content.lower()
            if any(drug in text_lower for drug in drug_set):
                base_score += drug_boost
                
            scored_docs.append((doc, base_score))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top chunks, prioritizing comparator coverage
        top_docs = [doc for doc, _ in scored_docs[:final_k*2]]  # Get more than needed initially
        
        # Prioritize by comparator coverage
        selected_docs, _, _ = self.deduplicator.prioritize_by_comparator_coverage(
            top_docs, 
            final_k=final_k
        )
        
        # Return the formatted results
        return [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in selected_docs
        ]

    def retrieve_pico_chunks(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks by country using vector similarity search.
        """
        results_by_country = {}

        for country in countries:
            chunks = self.vector_similarity_search(
                query=query,
                country=country,
                source_type=source_type,
                heading_keywords=heading_keywords,
                drug_keywords=drug_keywords,
                initial_k=initial_k,
                final_k=final_k
            )
            results_by_country[country] = chunks

        return results_by_country
    
    def diagnose_vectorstore(self, limit: int = 100):
        """
        Diagnose the vectorstore by checking available metadata fields and values.
        """
        try:
            result = self.chroma_collection.get(
                limit=limit,
                include=['metadatas']
            )
            
            print(f"=== VECTORSTORE DIAGNOSTICS ===")
            print(f"Total documents sampled: {len(result['metadatas'])}")
            
            # Analyze metadata fields
            all_fields = set()
            country_values = set()
            source_type_values = set()
            
            for metadata in result['metadatas']:
                if metadata:
                    all_fields.update(metadata.keys())
                    if 'country' in metadata:
                        country_values.add(metadata['country'])
                    if 'source_type' in metadata:
                        source_type_values.add(metadata['source_type'])
            
            print(f"Available metadata fields: {sorted(all_fields)}")
            print(f"Countries found: {sorted(country_values)}")
            print(f"Source types found: {sorted(source_type_values)}")
            
            # Show sample metadata
            print(f"\nSample metadata entries:")
            for i, metadata in enumerate(result['metadatas'][:5]):
                print(f"  {i+1}: {metadata}")
                
            return {
                'total_docs': len(result['metadatas']),
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
            
            # Test 1: No filters
            result_no_filter = self.chroma_collection.get(limit=limit)
            print(f"Documents without filter: {len(result_no_filter['documents'])}")
            
            # Test 2: Country filter only
            result_country = self.chroma_collection.get(
                where={"country": country},
                limit=limit
            )
            print(f"Documents with country={country}: {len(result_country['documents'])}")
            
            # Test 3: Try vector similarity without filters
            docs_no_filter = self.vectorstore.similarity_search(
                query="treatment",
                k=limit
            )
            print(f"Vector similarity without filter: {len(docs_no_filter)} docs")
            
            # Test 4: Try vector similarity with country filter
            docs_with_filter = self.vectorstore.similarity_search(
                query="treatment",
                k=limit,
                filter={"country": country}
            )
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
        initial_k: int = 30,
        final_k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Test the retrieval pipeline and return results with simple summary.
        """
        results = self.retrieve_pico_chunks(
            query=query,
            countries=countries,
            source_type=source_type,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k
        )
        
        # Print simple summary
        print(f"\n=== RETRIEVAL RESULTS ===")
        print(f"Query: {query}")
        print(f"Source type: {source_type or 'All sources'}")
        
        total_chunks = 0
        for country, chunks in results.items():
            chunk_count = len(chunks)
            total_chunks += chunk_count
            print(f"{country}: {chunk_count} chunks")
        
        print(f"Total: {total_chunks} chunks")
        print("=" * 25)
        
        return results


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
    """
    def __init__(
        self,
        chunk_retriever,
        system_prompt: str,
        user_prompt_template: str,
        model_name: str = "gpt-4o-mini"
    ):
        self.chunk_retriever = chunk_retriever
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.context_manager = ContextManager()

    def extract_picos(
        self,
        countries: List[str],
        query: str,
        source_type: Optional[str] = None,
        initial_k: int = 10,
        final_k: int = 5,
        heading_keywords: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs using improved context management and LLM.
        """
        results = []

        # Optionally override the model
        llm_to_use = self.llm
        if model_override:
            if isinstance(model_override, str):
                llm_to_use = ChatOpenAI(model_name=model_override, temperature=0)
            elif isinstance(model_override, ChatOpenAI):
                llm_to_use = model_override

        # Ensure the output directory exists
        os.makedirs("results", exist_ok=True)

        for country in countries:
            # Retrieve chunks for the country
            results_dict = self.chunk_retriever.retrieve_pico_chunks(
                query=query,
                countries=[country],
                source_type=source_type,
                heading_keywords=heading_keywords,
                initial_k=initial_k,
                final_k=final_k
            )
            
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

            # Save results
            source_prefix = source_type.replace("_", "") if source_type else "general"
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country  # Ensure correct country
                parsed_json["SourceType"] = source_type  # Add source type info
                results.append(parsed_json)
                outpath = os.path.join("results", f"{source_prefix}_picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            else:
                # Handle non-dict response
                wrapped_json = {
                    "Country": country,
                    "SourceType": source_type,
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                results.append(wrapped_json)
                outpath = os.path.join("results", f"{source_prefix}_picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(wrapped_json, f, indent=2, ensure_ascii=False)

        return results