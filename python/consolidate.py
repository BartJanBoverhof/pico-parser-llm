import json
import os
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
        self.consolidated_dir = os.path.join(results_output_dir, "consolidated")
        self.consolidation_configs = consolidation_configs or {}

        # Create consolidated output directory
        os.makedirs(self.consolidated_dir, exist_ok=True)

        # Initialize OpenAI client and ChatOpenAI
        self.openai_client = OpenAI()
        self.chat_model = ChatOpenAI(
            model=self.model_name,
            temperature=0.1
        )

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
            # Auto-detect available PICO files
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

    def extract_all_picos_and_outcomes(self, pico_data: Dict[str, Dict]) -> tuple:
        """
        Extract all PICOs and outcomes from the loaded data.

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

        for source_type, data in pico_data.items():
            all_source_types.add(source_type)
            
            # Extract metadata
            if indication is None:
                indication = data.get("extraction_metadata", {}).get("indication", "")

            # Process each country's PICOs
            for country, country_data in data.get("picos_by_country", {}).items():
                all_countries.add(country)
                
                for pico in country_data.get("PICOs", []):
                    # Add metadata to each PICO
                    pico_with_metadata = pico.copy()
                    pico_with_metadata["Country"] = country
                    pico_with_metadata["Source_Type"] = source_type
                    all_picos.append(pico_with_metadata)
                    
                    # Collect outcomes if present
                    if pico.get("Outcomes") and pico["Outcomes"].strip():
                        outcome_entry = {
                            "Outcomes": pico["Outcomes"],
                            "Country": country,
                            "Source_Type": source_type,
                            "Population": pico.get("Population", ""),
                            "Comparator": pico.get("Comparator", "")
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

        # Use model override if provided
        model = model_override if model_override else self.chat_model

        # Prepare system prompt
        system_prompt = self.consolidation_configs.get(
            "pico_consolidation_system_prompt", 
            "Consolidate the provided PICOs by grouping similar Population and Comparator combinations."
        )

        # Prepare the PICO data for the LLM
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

        # Create user prompt
        user_prompt = f"""
        Indication: {metadata.get('indication', '')}

        Source Countries: {', '.join(metadata.get('countries', []))}
        Source Types: {', '.join(metadata.get('source_types', []))}

        PICOs to consolidate:
        {json.dumps(picos_for_llm, indent=2)}

        Task: Consolidate these PICOs into a non-redundant list where PICOs with substantially similar Population and Comparator combinations are merged. Track the countries and source types for each consolidated PICO.

        Return the result as valid JSON following the structure specified in the system prompt.
        """

        # Call LLM for consolidation
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            if isinstance(model, str):
                # Use OpenAI directly
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                result_text = response.choices[0].message.content
            else:
                # Use ChatOpenAI
                response = model.invoke(messages)
                result_text = response.content

            # Parse the JSON response
            consolidated_result = json.loads(result_text)
            
            # Add timestamp and ensure metadata is complete
            consolidated_result["consolidation_metadata"]["timestamp"] = datetime.now().isoformat()
            consolidated_result["consolidation_metadata"]["indication"] = metadata.get("indication", "")
            consolidated_result["consolidation_metadata"]["source_countries"] = metadata.get("countries", [])
            consolidated_result["consolidation_metadata"]["source_types"] = metadata.get("source_types", [])
            
            return consolidated_result

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

        Args:
            all_outcomes: List of all outcome dictionaries
            metadata: Metadata about the data
            model_override: Optional model override

        Returns:
            Consolidated outcomes result dictionary
        """
        if not all_outcomes:
            print("No outcomes to consolidate")
            return {}

        # Use model override if provided
        model = model_override if model_override else self.chat_model

        # Prepare system prompt
        system_prompt = self.consolidation_configs.get(
            "outcomes_consolidation_system_prompt", 
            "Consolidate and categorize the provided outcomes into organized categories."
        )

        # Prepare the outcomes data for the LLM
        outcomes_for_llm = []
        for i, outcome in enumerate(all_outcomes):
            outcome_entry = {
                "id": i + 1,
                "Outcomes": outcome.get("Outcomes", ""),
                "Country": outcome.get("Country", ""),
                "Source_Type": outcome.get("Source_Type", ""),
                "Population_Context": outcome.get("Population", ""),
                "Comparator_Context": outcome.get("Comparator", "")
            }
            outcomes_for_llm.append(outcome_entry)

        # Create user prompt
        user_prompt = f"""
        Indication: {metadata.get('indication', '')}

        Source Countries: {', '.join(metadata.get('countries', []))}
        Source Types: {', '.join(metadata.get('source_types', []))}

        Outcomes to consolidate and categorize:
        {json.dumps(outcomes_for_llm, indent=2)}

        Task: Consolidate these outcomes into organized categories (Efficacy, Safety, Quality of Life, Economic, Other) with appropriate subcategories. Remove duplicates but preserve important measurement details and clinical context.

        Return the result as valid JSON following the structure specified in the system prompt.
        """

        # Call LLM for consolidation
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            if isinstance(model, str):
                # Use OpenAI directly
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                result_text = response.choices[0].message.content
            else:
                # Use ChatOpenAI
                response = model.invoke(messages)
                result_text = response.content

            # Parse the JSON response
            consolidated_result = json.loads(result_text)
            
            # Add timestamp and ensure metadata is complete
            consolidated_result["outcomes_metadata"]["timestamp"] = datetime.now().isoformat()
            consolidated_result["outcomes_metadata"]["indication"] = metadata.get("indication", "")
            consolidated_result["outcomes_metadata"]["source_countries"] = metadata.get("countries", [])
            consolidated_result["outcomes_metadata"]["source_types"] = metadata.get("source_types", [])
            
            return consolidated_result

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
        Save consolidated results to a single JSON file.

        Args:
            consolidated_picos: Consolidated PICO results
            consolidated_outcomes: Consolidated outcomes results
            indication: Indication string for filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create safe filename from indication
        safe_indication = "".join(c for c in indication if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_indication = safe_indication.replace(' ', '_').lower()
        if len(safe_indication) > 50:
            safe_indication = safe_indication[:50]
        
        if not safe_indication:
            safe_indication = "consolidated"

        # Combine both results into a single file
        if consolidated_picos or consolidated_outcomes:
            combined_results = {
                "consolidation_metadata": {
                    "timestamp": timestamp,
                    "indication": indication,
                    "consolidation_types": []
                }
            }
            
            # Add consolidated PICOs if available
            if consolidated_picos:
                combined_results["consolidated_picos"] = consolidated_picos.get("consolidated_picos", [])
                combined_results["consolidation_metadata"]["pico_metadata"] = consolidated_picos.get("consolidation_metadata", {})
                combined_results["consolidation_metadata"]["consolidation_types"].append("picos")
            
            # Add consolidated outcomes if available
            if consolidated_outcomes:
                combined_results["consolidated_outcomes"] = consolidated_outcomes.get("outcomes_by_category", {})
                combined_results["consolidation_metadata"]["outcomes_metadata"] = consolidated_outcomes.get("outcomes_metadata", {})
                combined_results["consolidation_metadata"]["consolidation_types"].append("outcomes")
            
            # Save combined results to single file
            consolidated_filename = f"{safe_indication}_consolidated_{timestamp}.json"
            consolidated_filepath = os.path.join(self.consolidated_dir, consolidated_filename)
            
            try:
                with open(consolidated_filepath, 'w', encoding='utf-8') as f:
                    json.dump(combined_results, f, indent=2, ensure_ascii=False)
                print(f"Saved consolidated results to: {consolidated_filepath}")
                return consolidated_filepath
            except Exception as e:
                print(f"Error saving consolidated results: {e}")
                return None
        
        return None

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
        
        # Step 1: Load PICO files
        print("Step 1: Loading PICO extraction files...")
        pico_data = self.load_pico_files(source_types)
        
        if not pico_data:
            print("No PICO data found to consolidate")
            return {}

        # Step 2: Extract all PICOs and outcomes
        print("Step 2: Extracting PICOs and outcomes from loaded data...")
        all_picos, all_outcomes, metadata = self.extract_all_picos_and_outcomes(pico_data)
        
        print(f"Found {len(all_picos)} total PICOs across {len(metadata['countries'])} countries")
        print(f"Found {len(all_outcomes)} total outcomes")

        # Step 3: Consolidate PICOs
        print("Step 3: Consolidating PICOs...")
        consolidated_picos = self.consolidate_picos(all_picos, metadata, model_override)
        
        if consolidated_picos:
            total_consolidated = consolidated_picos.get("consolidation_metadata", {}).get("total_consolidated_picos", 0)
            print(f"Consolidated {len(all_picos)} original PICOs into {total_consolidated} consolidated PICOs")

        # Step 4: Consolidate outcomes
        print("Step 4: Consolidating outcomes...")
        consolidated_outcomes = self.consolidate_outcomes(all_outcomes, metadata, model_override)
        
        if consolidated_outcomes:
            total_unique = consolidated_outcomes.get("outcomes_metadata", {}).get("total_unique_outcomes", 0)
            print(f"Organized {len(all_outcomes)} original outcomes into {total_unique} unique categorized outcomes")

        # Step 5: Save results
        print("Step 5: Saving consolidated results...")
        self.save_consolidated_results(
            consolidated_picos, 
            consolidated_outcomes, 
            metadata.get("indication", "")
        )

        print("=== Consolidation Complete ===")
        
        return {
            "consolidated_picos": consolidated_picos,
            "consolidated_outcomes": consolidated_outcomes,
            "metadata": metadata,
            "summary": {
                "original_picos": len(all_picos),
                "consolidated_picos": consolidated_picos.get("consolidation_metadata", {}).get("total_consolidated_picos", 0),
                "original_outcomes": len(all_outcomes),
                "unique_outcomes": consolidated_outcomes.get("outcomes_metadata", {}).get("total_unique_outcomes", 0),
                "countries": metadata.get("countries", []),
                "source_types": metadata.get("source_types", [])
            }
        }