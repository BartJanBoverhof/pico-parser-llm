import os
import json
from typing import List, Dict, Any, Optional, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from datetime import datetime


class PICOExtractor:
    """
    PICOExtractor with improved chunk deduplication and adaptive context handling.
    Enhanced with JSON storage for organized PICO results and indication parameterization.
    """
    def __init__(
        self,
        system_prompt: str,
        user_prompt_template: str,
        model_name: str = "gpt-4o-mini",
        results_output_dir: str = "results"
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.context_manager = ContextManager()
        self.results_output_dir = results_output_dir
        self.chunks_input_dir = os.path.join(results_output_dir, "chunks")
        self.pico_output_dir = os.path.join(results_output_dir, "PICO")
        os.makedirs(self.pico_output_dir, exist_ok=True)

    def load_chunks_from_file(self, filename: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load chunks from a saved retrieval results file.
        """
        filepath = os.path.join(self.chunks_input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Chunks file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results_by_country = {}
            if "results_by_country" in data:
                for country, country_data in data["results_by_country"].items():
                    results_by_country[country] = country_data.get("chunks", [])
            
            return results_by_country
        except Exception as e:
            print(f"Error loading chunks from {filepath}: {e}")
            return {}

    def extract_picos_from_chunks(
        self,
        chunks_by_country: Dict[str, List[Dict[str, Any]]],
        source_type: Optional[str] = None,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs from pre-loaded chunks using LLM.
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

        # Process each country
        for country, country_chunks in chunks_by_country.items():
            if not country_chunks:
                print(f"No chunks available for {country}")
                continue

            # Process chunks with context manager
            processed_chunks = self.context_manager.process_chunks(country_chunks)
            context_block = self.context_manager.build_optimal_context(processed_chunks)

            # Prepare system and user messages with indication parameterization
            system_msg = SystemMessage(content=self.system_prompt)
            if indication:
                user_msg_text = self.user_prompt_template.format(
                    indication=indication,
                    context_block=context_block,
                    country=country
                )
            else:
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

        return results

    def extract_picos(
        self,
        source_type: str,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs by loading chunks from stored retrieval results.
        """
        # Determine filename based on source_type and indication
        if indication:
            indication_short = indication.split()[0].lower() if indication else "unknown"
            filename = f"{source_type}_{indication_short}_retrieval_results.json"
        else:
            filename = f"{source_type}_retrieval_results.json"
        
        # Load chunks from file
        chunks_by_country = self.load_chunks_from_file(filename)
        
        if not chunks_by_country:
            print(f"No chunks loaded for {source_type} with indication {indication}")
            return []
        
        # Extract PICOs from loaded chunks
        results = self.extract_picos_from_chunks(
            chunks_by_country=chunks_by_country,
            source_type=source_type,
            indication=indication,
            model_override=model_override
        )
        
        # Save organized PICO results to JSON
        if results:
            self._save_pico_results_with_indication(results, source_type, indication, datetime.now().isoformat())
        
        return results

    def extract_picos_with_indication(
        self,
        source_type: str,
        indication: str,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs with parameterized indication by loading chunks from stored retrieval results.
        """
        return self.extract_picos(
            source_type=source_type,
            indication=indication,
            model_override=model_override
        )

    def _save_pico_results_with_indication(self, results: List[Dict[str, Any]], source_type: str, indication: Optional[str], timestamp: str):
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
        if indication:
            indication_short = indication.split()[0].lower() if indication else "unknown"
            filename = f"{source_prefix}_{indication_short}_picos_organized.json"
        else:
            filename = f"{source_prefix}_picos_organized.json"
        filepath = os.path.join(self.pico_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved organized PICO results to {filepath}")
        return filepath