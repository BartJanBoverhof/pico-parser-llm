import json
import os
import tiktoken
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class PICOConsolidator:
    """
    A class to consolidate PICOs and outcomes from multiple countries and source types
    into unified, non-redundant lists organized by clinical relevance.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        results_output_dir: str = "results",
        consolidation_configs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the PICO consolidator.

        Args:
            model_name: OpenAI model to use for consolidation
            results_output_dir: Directory containing PICO extraction results
            consolidation_configs: Configuration for consolidation prompts
        """
        self.model_name = model_name
        self.results_dir = results_output_dir
        self.pico_dir = os.path.join(results_output_dir, "PICO")
        self.chunks_dir = os.path.join(results_output_dir, "chunks")
        self.consolidated_dir = os.path.join(results_output_dir, "consolidated")
        self.consolidation_configs = consolidation_configs or {}

        os.makedirs(self.consolidated_dir, exist_ok=True)

        self.openai_client = OpenAI()
        self.chat_model = ChatOpenAI(
            model=self.model_name,
            temperature=0.1
        )
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the current encoding."""
        return len(self.encoding.encode(text))

    def load_pico_files(self, source_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Load PICO extraction results from JSON files.

        Args:
            source_types: List of source types to load. If None, loads all available.

        Returns:
            Dictionary with source types as keys and loaded data as values
        """
        if source_types is None:
            source_types = []
            for file in os.listdir(self.pico_dir):
                if file.endswith("_picos.json"):
                    source_type = file.replace("_picos.json", "")
                    source_types.append(source_type)

        pico_data = {}
        for source_type in source_types:
            file_path = os.path.join(self.pico_dir, f"{source_type}_picos.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        pico_data[source_type] = json.load(f)
                    print(f"Loaded PICOs from {source_type}: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"PICO file not found: {file_path}")

        return pico_data

    def load_separate_outcome_extractions(self, source_types: List[str]) -> Dict[str, Dict]:
        """
        Load separate outcome extraction results if they exist.

        Args:
            source_types: List of source types to look for

        Returns:
            Dictionary with source types as keys and outcome data as values
        """
        outcome_data = {}
        
        for source_type in source_types:
            outcome_files = []
            if os.path.exists(self.chunks_dir):
                for file in os.listdir(self.chunks_dir):
                    if f"{source_type}_outcomes" in file and file.endswith("_extraction_results.json"):
                        outcome_files.append(file)
            
            if outcome_files:
                print(f"Found separate outcome extraction files for {source_type}: {outcome_files}")
                outcome_files.sort(reverse=True)
                outcome_file_path = os.path.join(self.chunks_dir, outcome_files[0])
                
                try:
                    with open(outcome_file_path, 'r', encoding='utf-8') as f:
                        outcome_data[source_type] = json.load(f)
                    print(f"Loaded separate outcomes for {source_type}: {outcome_file_path}")
                except Exception as e:
                    print(f"Error loading separate outcomes {outcome_file_path}: {e}")
        
        return outcome_data

    def combine_picos_with_outcomes(self, pico_data: Dict[str, Dict], outcome_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Combine PICO data with separate outcome extractions.

        Args:
            pico_data: PICO extraction data
            outcome_data: Separate outcome extraction data

        Returns:
            Combined PICO data with outcomes filled in
        """
        combined_data = {}
        
        for source_type, picos in pico_data.items():
            combined_data[source_type] = picos.copy()
            
            if source_type in outcome_data:
                outcomes_by_country = {}
                
                outcome_source_data = outcome_data[source_type]
                if "outcomes_by_country" in outcome_source_data:
                    for country, country_outcome_data in outcome_source_data["outcomes_by_country"].items():
                        if "Outcomes" in country_outcome_data:
                            outcomes_by_country[country] = country_outcome_data["Outcomes"]
                
                if outcomes_by_country and "picos_by_country" in combined_data[source_type]:
                    for country, country_pico_data in combined_data[source_type]["picos_by_country"].items():
                        if country in outcomes_by_country:
                            country_outcomes = outcomes_by_country[country]
                            for pico in country_pico_data.get("PICOs", []):
                                if not pico.get("Outcomes") or pico["Outcomes"].strip() == "":
                                    pico["Outcomes"] = country_outcomes
                            
                            print(f"Applied outcomes to {len(country_pico_data.get('PICOs', []))} PICOs for {source_type} {country}")
        
        return combined_data

    def extract_all_picos_and_outcomes(self, pico_data: Dict[str, Dict]) -> tuple:
        """
        Extract all PICOs and outcomes from the loaded data.
        For outcomes, extract one outcome list per country/source rather than per PICO.

        Args:
            pico_data: Dictionary containing loaded PICO data by source type

        Returns:
            Tuple of (all_picos_list, all_outcomes_list, metadata)
        """
        all_picos = []
        all_outcomes = []
        all_countries = set()
        all_source_types = set()
        indication = None
        
        country_outcomes_seen = set()

        for source_type, data in pico_data.items():
            all_source_types.add(source_type)
            
            if indication is None:
                indication = data.get("extraction_metadata", {}).get("indication", "")

            for country, country_data in data.get("picos_by_country", {}).items():
                all_countries.add(country)
                
                for pico in country_data.get("PICOs", []):
                    pico_with_metadata = pico.copy()
                    pico_with_metadata["Country"] = country
                    pico_with_metadata["Source_Type"] = source_type
                    all_picos.append(pico_with_metadata)
                
                country_source_key = f"{country}_{source_type}"
                if country_source_key not in country_outcomes_seen:
                    country_outcomes_seen.add(country_source_key)
                    
                    first_pico = country_data.get("PICOs", [{}])[0]
                    if first_pico.get("Outcomes") and first_pico["Outcomes"].strip():
                        outcome_entry = {
                            "Outcomes": first_pico["Outcomes"],
                            "Country": country,
                            "Source_Type": source_type
                        }
                        all_outcomes.append(outcome_entry)

        metadata = {
            "indication": indication,
            "countries": sorted(list(all_countries)),
            "source_types": sorted(list(all_source_types)),
            "total_original_picos": len(all_picos),
            "total_original_outcomes": len(all_outcomes)
        }

        return all_picos, all_outcomes, metadata

    def clean_llm_response(self, response_text: str) -> str:
        """
        Clean LLM response to extract valid JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        if not response_text:
            return ""
        
        response_text = response_text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end]
        
        return response_text

    def consolidate_picos(
        self, 
        all_picos: List[Dict], 
        metadata: Dict,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict:
        """
        Consolidate PICOs using LLM to merge similar Population-Comparator combinations.

        Args:
            all_picos: List of all PICO dictionaries
            metadata: Metadata about the data
            model_override: Optional model override

        Returns:
            Consolidated PICOs result dictionary
        """
        if not all_picos:
            print("No PICOs to consolidate")
            return {}

        model = model_override if model_override else self.chat_model

        system_prompt = self.consolidation_configs.get(
            "pico_consolidation_system_prompt", 
            "Consolidate the provided PICOs by grouping similar Population and Comparator combinations."
        )

        picos_for_llm = []
        for i, pico in enumerate(all_picos):
            pico_entry = {
                "id": i + 1,
                "Population": pico.get("Population", ""),
                "Intervention": pico.get("Intervention", ""),
                "Comparator": pico.get("Comparator", ""),
                "Country": pico.get("Country", ""),
                "Source_Type": pico.get("Source_Type", "")
            }
            picos_for_llm.append(pico_entry)

        user_prompt = f"""
        Indication: {metadata.get('indication', '')}

        Source Countries: {', '.join(metadata.get('countries', []))}
        Source Types: {', '.join(metadata.get('source_types', []))}

        PICOs to consolidate:
        {json.dumps(picos_for_llm, indent=2)}

        Task: Consolidate these PICOs into a non-redundant list where PICOs with substantially similar Population and Comparator combinations are merged. Track the countries and source types for each consolidated PICO.

        Return the result as valid JSON following the structure specified in the system prompt.
        """

        input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            if isinstance(model, str):
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                result_text = response.choices[0].message.content
            else:
                response = model.invoke(messages)
                result_text = response.content

            output_tokens = self.count_tokens(result_text) if result_text else 0
            print(f"PICO consolidation - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            
            print(f"PICO consolidation LLM response length: {len(result_text) if result_text else 0}")
            print(f"PICO consolidation LLM response preview: {result_text[:200] if result_text else 'None'}")
            
            if not result_text or result_text.strip() == "":
                print("Empty response from LLM for PICO consolidation")
                return {}
            
            cleaned_text = self.clean_llm_response(result_text)
            
            if not cleaned_text:
                print("No valid JSON found in PICO consolidation response")
                return {}

            consolidated_result = json.loads(cleaned_text)
            
            if "consolidation_metadata" not in consolidated_result:
                consolidated_result["consolidation_metadata"] = {}
            
            consolidated_result["consolidation_metadata"]["timestamp"] = datetime.now().isoformat()
            consolidated_result["consolidation_metadata"]["indication"] = metadata.get("indication", "")
            consolidated_result["consolidation_metadata"]["source_countries"] = metadata.get("countries", [])
            consolidated_result["consolidation_metadata"]["source_types"] = metadata.get("source_types", [])
            
            total_consolidated = len(consolidated_result.get("consolidated_picos", []))
            print(f"Successfully consolidated {len(all_picos)} original PICOs into {total_consolidated} consolidated PICOs")
            
            return consolidated_result

        except json.JSONDecodeError as e:
            print(f"JSON parsing error during PICO consolidation: {e}")
            print(f"Raw response: {result_text[:500] if result_text else 'None'}...")
            return {}
        except Exception as e:
            print(f"Error during PICO consolidation: {e}")
            return {}

    def consolidate_outcomes(
        self, 
        all_outcomes: List[Dict], 
        metadata: Dict,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict:
        """
        Consolidate outcomes using LLM to organize into structured categories.
        Outcomes are consolidated from multiple countries into unified categories.

        Args:
            all_outcomes: List of outcome dictionaries (one per country/source)
            metadata: Metadata about the data
            model_override: Optional model override

        Returns:
            Consolidated outcomes result dictionary
        """
        if not all_outcomes:
            print("No outcomes to consolidate")
            return {}

        model = model_override if model_override else self.chat_model

        system_prompt = self.consolidation_configs.get(
            "outcomes_consolidation_system_prompt", 
            "Consolidate and categorize the provided outcomes into organized categories."
        )

        outcomes_for_llm = []
        for i, outcome in enumerate(all_outcomes):
            outcome_entry = {
                "id": i + 1,
                "Outcomes": outcome.get("Outcomes", ""),
                "Country": outcome.get("Country", ""),
                "Source_Type": outcome.get("Source_Type", "")
            }
            outcomes_for_llm.append(outcome_entry)

        user_prompt = f"""
        Indication: {metadata.get('indication', '')}

        Source Countries: {', '.join(metadata.get('countries', []))}
        Source Types: {', '.join(metadata.get('source_types', []))}

        Outcomes to consolidate and categorize (one list per country/source):
        {json.dumps(outcomes_for_llm, indent=2)}

        Task: Consolidate these outcome lists into organized categories (Efficacy, Safety, Quality of Life, Economic, Other) with appropriate subcategories. Remove duplicates but preserve important measurement details and clinical context. Each country/source provides one comprehensive outcomes list.

        Return the result as valid JSON following the structure specified in the system prompt.
        """

        input_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            if isinstance(model, str):
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                result_text = response.choices[0].message.content
            else:
                response = model.invoke(messages)
                result_text = response.content

            output_tokens = self.count_tokens(result_text) if result_text else 0
            print(f"Outcomes consolidation - Input tokens: {input_tokens}, Output tokens: {output_tokens}")

            print(f"Outcomes consolidation LLM response length: {len(result_text) if result_text else 0}")
            print(f"Outcomes consolidation LLM response preview: {result_text[:200] if result_text else 'None'}")
            
            if not result_text or result_text.strip() == "":
                print("Empty response from LLM for outcomes consolidation")
                return {}
            
            cleaned_text = self.clean_llm_response(result_text)
            
            if not cleaned_text:
                print("No valid JSON found in outcomes consolidation response")
                return {}

            consolidated_result = json.loads(cleaned_text)
            
            if "outcomes_metadata" not in consolidated_result:
                consolidated_result["outcomes_metadata"] = {}
            
            consolidated_result["outcomes_metadata"]["timestamp"] = datetime.now().isoformat()
            consolidated_result["outcomes_metadata"]["indication"] = metadata.get("indication", "")
            consolidated_result["outcomes_metadata"]["source_countries"] = metadata.get("countries", [])
            consolidated_result["outcomes_metadata"]["source_types"] = metadata.get("source_types", [])
            
            total_unique = consolidated_result.get("outcomes_metadata", {}).get("total_unique_outcomes", 0)
            print(f"Successfully organized {len(all_outcomes)} original outcome lists into {total_unique} unique categorized outcomes")
            
            return consolidated_result

        except json.JSONDecodeError as e:
            print(f"JSON decode error during outcomes consolidation: {e}")
            print(f"Raw LLM response: {result_text[:1000] if result_text else 'None'}")
            return {}
        except Exception as e:
            print(f"Error during outcomes consolidation: {e}")
            return {}

    def save_consolidated_results(
        self, 
        consolidated_picos: Dict, 
        consolidated_outcomes: Dict,
        indication: str = ""
    ):
        """
        Save consolidated results to separate JSON files.

        Args:
            consolidated_picos: Consolidated PICO results
            consolidated_outcomes: Consolidated outcomes results
            indication: Indication string for filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        safe_indication = "".join(c for c in indication if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_indication = safe_indication.replace(' ', '_').lower()
        if len(safe_indication) > 50:
            safe_indication = safe_indication[:50]
        
        if not safe_indication:
            safe_indication = "consolidated"

        saved_files = {}

        if consolidated_picos and isinstance(consolidated_picos, dict) and consolidated_picos.get("consolidated_picos"):
            picos_result = {
                "consolidation_metadata": {
                    "timestamp": timestamp,
                    "indication": indication,
                    "consolidation_type": "picos"
                },
                "consolidated_picos": consolidated_picos.get("consolidated_picos", [])
            }
            
            if "consolidation_metadata" in consolidated_picos:
                picos_result["consolidation_metadata"].update(consolidated_picos["consolidation_metadata"])
            
            picos_filename = f"{safe_indication}_consolidated_picos_{timestamp}.json"
            picos_filepath = os.path.join(self.consolidated_dir, picos_filename)
            
            try:
                with open(picos_filepath, 'w', encoding='utf-8') as f:
                    json.dump(picos_result, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(picos_result['consolidated_picos'])} consolidated PICOs to: {picos_filepath}")
                saved_files["picos_file"] = picos_filepath
            except Exception as e:
                print(f"Error saving consolidated PICOs: {e}")
        else:
            print("No consolidated PICOs to save (consolidation may have failed)")

        if consolidated_outcomes and isinstance(consolidated_outcomes, dict) and consolidated_outcomes.get("outcomes_by_category"):
            outcomes_result = {
                "outcomes_metadata": {
                    "timestamp": timestamp,
                    "indication": indication,
                    "consolidation_type": "outcomes"
                },
                "consolidated_outcomes": consolidated_outcomes.get("outcomes_by_category", {})
            }
            
            if "outcomes_metadata" in consolidated_outcomes:
                outcomes_result["outcomes_metadata"].update(consolidated_outcomes["outcomes_metadata"])
            
            outcomes_filename = f"{safe_indication}_consolidated_outcomes_{timestamp}.json"
            outcomes_filepath = os.path.join(self.consolidated_dir, outcomes_filename)
            
            try:
                with open(outcomes_filepath, 'w', encoding='utf-8') as f:
                    json.dump(outcomes_result, f, indent=2, ensure_ascii=False)
                
                total_outcomes = 0
                for category in outcomes_result.get("consolidated_outcomes", {}).values():
                    if isinstance(category, dict):
                        for subcategory in category.values():
                            if isinstance(subcategory, list):
                                total_outcomes += len(subcategory)
                
                print(f"Saved {total_outcomes} consolidated outcomes to: {outcomes_filepath}")
                saved_files["outcomes_file"] = outcomes_filepath
            except Exception as e:
                print(f"Error saving consolidated outcomes: {e}")
        else:
            print("No consolidated outcomes to save (consolidation may have failed)")
        
        if not saved_files:
            print("Warning: No consolidated files were saved. Check consolidation process.")
        
        return saved_files

    def consolidate_all(
        self, 
        source_types: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict:
        """
        Run the complete consolidation pipeline.

        Args:
            source_types: List of source types to consolidate
            model_override: Optional model override

        Returns:
            Dictionary with consolidation results
        """
        print("=== Starting PICO and Outcomes Consolidation ===")
        
        print("Step 1: Loading PICO extraction files...")
        pico_data = self.load_pico_files(source_types)
        
        if not pico_data:
            print("No PICO data found to consolidate")
            return {}

        print("Step 1.5: Checking for separate outcome extractions...")
        outcome_data = self.load_separate_outcome_extractions(list(pico_data.keys()))
        
        if outcome_data:
            print("Found separate outcome extractions, combining with PICOs...")
            pico_data = self.combine_picos_with_outcomes(pico_data, outcome_data)
        else:
            print("No separate outcome extractions found, proceeding with existing PICO data")

        print("Step 2: Extracting PICOs and outcomes from loaded data...")
        all_picos, all_outcomes, metadata = self.extract_all_picos_and_outcomes(pico_data)
        
        print(f"Found {len(all_picos)} total PICOs across {len(metadata['countries'])} countries")
        print(f"Found {len(all_outcomes)} total outcome lists (one per country/source)")

        print("Step 3: Consolidating PICOs...")
        consolidated_picos = self.consolidate_picos(all_picos, metadata, model_override)

        print("Step 4: Consolidating outcomes...")
        consolidated_outcomes = self.consolidate_outcomes(all_outcomes, metadata, model_override)

        print("Step 5: Saving consolidated results...")
        saved_files = self.save_consolidated_results(
            consolidated_picos, 
            consolidated_outcomes, 
            metadata.get("indication", "")
        )

        print("=== Consolidation Complete ===")
        
        return {
            "consolidated_picos": consolidated_picos,
            "consolidated_outcomes": consolidated_outcomes,
            "metadata": metadata,
            "saved_files": saved_files,
            "summary": {
                "original_picos": len(all_picos),
                "consolidated_picos": len(consolidated_picos.get("consolidated_picos", [])) if consolidated_picos else 0,
                "original_outcomes": len(all_outcomes),
                "unique_outcomes": consolidated_outcomes.get("outcomes_metadata", {}).get("total_unique_outcomes", 0) if consolidated_outcomes else 0,
                "countries": metadata.get("countries", []),
                "source_types": metadata.get("source_types", [])
            }
        }