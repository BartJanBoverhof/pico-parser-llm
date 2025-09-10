import os
import json
import tiktoken
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class PICOExtractor:
    """
    PICO extractor that uses split extraction pipeline.
    Extracts Population & Comparator separately from Outcomes, then combines results.
    """
    def __init__(
        self,
        system_prompt: str,
        user_prompt_template: str,
        source_type: str,
        model_name: str = "gpt-4o-mini",
        results_output_dir: str = "results",
        max_tokens: int = 12000,
        source_type_config: Optional[Dict[str, Any]] = None
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.source_type = source_type
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.results_output_dir = results_output_dir
        self.chunks_input_dir = os.path.join(results_output_dir, "chunks")
        self.pico_output_dir = os.path.join(results_output_dir, "PICO")
        self.source_type_config = source_type_config or {}
        
        os.makedirs(self.pico_output_dir, exist_ok=True)
        
        self.openai = OpenAI()
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the current encoding."""
        return len(self.encoding.encode(text))

    def find_chunk_file(self, source_type: str, retrieval_type: str = None, indication: Optional[str] = None) -> Optional[str]:
        """
        Find the appropriate chunk file based on source_type, retrieval_type and indication.
        """
        if not os.path.exists(self.chunks_input_dir):
            print(f"Chunks directory not found: {self.chunks_input_dir}")
            return None
        
        files = os.listdir(self.chunks_input_dir)
        chunk_files = [f for f in files if f.endswith('_retrieval_results.json')]
        
        if retrieval_type:
            if indication:
                indication_short = indication.split()[0].lower() if indication else "unknown"
                target_filename = f"{source_type}_{retrieval_type}_{indication_short}_retrieval_results.json"
                if target_filename in chunk_files:
                    return os.path.join(self.chunks_input_dir, target_filename)
            
            target_filename = f"{source_type}_{retrieval_type}_retrieval_results.json"
            if target_filename in chunk_files:
                return os.path.join(self.chunks_input_dir, target_filename)
        else:
            if indication:
                indication_short = indication.split()[0].lower() if indication else "unknown"
                target_filename = f"{source_type}_{indication_short}_retrieval_results.json"
                if target_filename in chunk_files:
                    return os.path.join(self.chunks_input_dir, target_filename)
            
            source_filename = f"{source_type}_retrieval_results.json"
            if source_filename in chunk_files:
                return os.path.join(self.chunks_input_dir, source_filename)
        
        for filename in chunk_files:
            if retrieval_type and source_type in filename and retrieval_type in filename:
                return os.path.join(self.chunks_input_dir, filename)
            elif not retrieval_type and source_type in filename:
                return os.path.join(self.chunks_input_dir, filename)
        
        print(f"No chunk file found for source_type: {source_type}, retrieval_type: {retrieval_type}, indication: {indication}")
        print(f"Available files: {chunk_files}")
        return None

    def load_chunks_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load chunks from a JSON file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading chunks from {file_path}: {e}")
            return {}

    def build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build optimal context from chunks while respecting token limits.
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue
            
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > self.max_tokens:
                if current_tokens == 0:
                    context_parts.append(chunk_text[:self.max_tokens * 3])
                    break
                else:
                    break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)

    def get_example_comparator(self, source_type: str) -> str:
        """
        Get an appropriate example comparator based on source type.
        """
        if source_type == "hta_submission":
            return "standard of care therapy"
        elif source_type == "clinical_guideline":
            return "recommended alternative therapy"
        else:
            return "appropriate comparator"

    def extract_population_comparator_from_context(
        self,
        context: str,
        indication: str,
        source_type: str = "hta_submission",
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict[str, Any]:
        """
        Extract Population and Comparator information from context.
        """
        if not context.strip():
            return {
                "Indication": indication,
                "Country": None,
                "PICOs": []
            }
        
        try:
            system_prompt = self.source_type_config.get("population_comparator_system_prompt", self.system_prompt)
            user_prompt_template = self.source_type_config.get("population_comparator_user_prompt_template", self.user_prompt_template)
            
            example_comparator = self.get_example_comparator(source_type)
            
            user_prompt = user_prompt_template.format(
                indication=indication,
                example_comparator=example_comparator,
                context_block=context
            )
            
            if model_override and isinstance(model_override, ChatOpenAI):
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = model_override.invoke(messages)
                result_text = response.content
            else:
                model_to_use = model_override if isinstance(model_override, str) else self.model_name
                
                response = self.openai.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content
            
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response for Population+Comparator. Raw response: {result_text[:500]}...")
                return {
                    "Indication": indication,
                    "Country": None,
                    "PICOs": [],
                    "Error": "JSON parsing failed",
                    "RawResponse": result_text
                }
                
        except Exception as e:
            print(f"Error in Population+Comparator extraction: {e}")
            return {
                "Indication": indication,
                "Country": None,
                "PICOs": [],
                "Error": str(e)
            }

    def extract_outcomes_from_context(
        self,
        context: str,
        indication: str,
        source_type: str = "hta_submission",
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict[str, Any]:
        """
        Extract Outcomes information from context.
        """
        if not context.strip():
            return {
                "Indication": indication,
                "Country": None,
                "Outcomes": ""
            }
        
        try:
            system_prompt = self.source_type_config.get("outcomes_system_prompt", "")
            user_prompt_template = self.source_type_config.get("outcomes_user_prompt_template", "")
            
            if not system_prompt or not user_prompt_template:
                print(f"Missing outcomes extraction prompts for source_type: {source_type}")
                return {
                    "Indication": indication,
                    "Country": None,
                    "Outcomes": "",
                    "Error": "Missing outcomes extraction prompts"
                }
            
            user_prompt = user_prompt_template.format(
                indication=indication,
                context_block=context
            )
            
            if model_override and isinstance(model_override, ChatOpenAI):
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = model_override.invoke(messages)
                result_text = response.content
            else:
                model_to_use = model_override if isinstance(model_override, str) else self.model_name
                
                response = self.openai.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content
            
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response for Outcomes. Raw response: {result_text[:500]}...")
                return {
                    "Indication": indication,
                    "Country": None,
                    "Outcomes": "",
                    "Error": "JSON parsing failed",
                    "RawResponse": result_text
                }
                
        except Exception as e:
            print(f"Error in Outcomes extraction: {e}")
            return {
                "Indication": indication,
                "Country": None,
                "Outcomes": "",
                "Error": str(e)
            }

    def extract_population_comparator(
        self,
        source_type: Optional[str] = None,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract Population and Comparator information from population_comparator chunks.
        """
        source_type_to_use = source_type or self.source_type
        
        chunk_file_path = self.find_chunk_file(source_type_to_use, "population_comparator", indication)
        if not chunk_file_path:
            print(f"No population_comparator chunk file found for source_type: {source_type_to_use}")
            return []
        
        print(f"Loading population_comparator chunks from: {chunk_file_path}")
        chunk_data = self.load_chunks_from_file(chunk_file_path)
        
        if not chunk_data:
            print("No population_comparator chunk data loaded")
            return []
        
        results_by_country = chunk_data.get("results_by_country", {})
        indication_from_metadata = chunk_data.get("retrieval_metadata", {}).get("indication") or indication or "unknown indication"
        
        extracted_results = []
        
        for country, country_data in results_by_country.items():
            chunks = country_data.get("chunks", [])
            
            if not chunks:
                print(f"No population_comparator chunks found for country: {country}")
                continue
            
            print(f"Processing {len(chunks)} population_comparator chunks for {country}")
            
            context = self.build_context_from_chunks(chunks)
            if not context:
                print(f"No population_comparator context built for country: {country}")
                continue
            
            pc_result = self.extract_population_comparator_from_context(
                context=context,
                indication=indication_from_metadata,
                source_type=source_type_to_use,
                model_override=model_override
            )
            
            pc_result["Country"] = country
            pc_result["ChunksUsed"] = len(chunks)
            pc_result["ContextTokens"] = self.count_tokens(context)
            
            extracted_results.append(pc_result)
            
            print(f"Extracted {len(pc_result.get('PICOs', []))} Population+Comparator entries for {country}")
        
        return extracted_results

    def extract_outcomes(
        self,
        source_type: Optional[str] = None,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract Outcomes information from outcomes chunks.
        """
        source_type_to_use = source_type or self.source_type
        
        chunk_file_path = self.find_chunk_file(source_type_to_use, "outcomes", indication)
        if not chunk_file_path:
            print(f"No outcomes chunk file found for source_type: {source_type_to_use}")
            return []
        
        print(f"Loading outcomes chunks from: {chunk_file_path}")
        chunk_data = self.load_chunks_from_file(chunk_file_path)
        
        if not chunk_data:
            print("No outcomes chunk data loaded")
            return []
        
        results_by_country = chunk_data.get("results_by_country", {})
        indication_from_metadata = chunk_data.get("retrieval_metadata", {}).get("indication") or indication or "unknown indication"
        
        extracted_results = []
        
        for country, country_data in results_by_country.items():
            chunks = country_data.get("chunks", [])
            
            if not chunks:
                print(f"No outcomes chunks found for country: {country}")
                continue
            
            print(f"Processing {len(chunks)} outcomes chunks for {country}")
            
            context = self.build_context_from_chunks(chunks)
            if not context:
                print(f"No outcomes context built for country: {country}")
                continue
            
            outcomes_result = self.extract_outcomes_from_context(
                context=context,
                indication=indication_from_metadata,
                source_type=source_type_to_use,
                model_override=model_override
            )
            
            outcomes_result["Country"] = country
            outcomes_result["ChunksUsed"] = len(chunks)
            outcomes_result["ContextTokens"] = self.count_tokens(context)
            
            extracted_results.append(outcomes_result)
            
            outcomes_text = outcomes_result.get('Outcomes') or ''
            print(f"Extracted outcomes for {country}: {outcomes_text[:100]}...")
        
        return extracted_results

    def combine_split_results(
        self,
        population_comparator_results: List[Dict[str, Any]],
        outcomes_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine Population+Comparator results with Outcomes results to create complete PICO entries.
        """
        combined_results = []
        
        pc_by_country = {result["Country"]: result for result in population_comparator_results}
        outcomes_by_country = {result["Country"]: result for result in outcomes_results}
        
        all_countries = set(pc_by_country.keys()) | set(outcomes_by_country.keys())
        
        for country in all_countries:
            pc_result = pc_by_country.get(country, {"PICOs": [], "Country": country, "ChunksUsed": 0, "ContextTokens": 0})
            outcomes_result = outcomes_by_country.get(country, {"Outcomes": "", "Country": country, "ChunksUsed": 0, "ContextTokens": 0})
            
            pc_picos = pc_result.get("PICOs", [])
            country_outcomes = outcomes_result.get("Outcomes") or ""
            
            if not pc_picos and not country_outcomes:
                continue
            
            if pc_picos:
                for pico in pc_picos:
                    pico["Outcomes"] = country_outcomes
            else:
                pc_picos = [{
                    "Population": outcomes_result.get("Indication", ""),
                    "Intervention": "Medicine X (under assessment)",
                    "Comparator": "",
                    "Outcomes": country_outcomes
                }]
            
            combined_result = {
                "Indication": pc_result.get("Indication") or outcomes_result.get("Indication", ""),
                "Country": country,
                "PICOs": pc_picos,
                "ChunksUsed": pc_result.get("ChunksUsed", 0) + outcomes_result.get("ChunksUsed", 0),
                "ContextTokens": pc_result.get("ContextTokens", 0) + outcomes_result.get("ContextTokens", 0)
            }
            
            combined_results.append(combined_result)
            
            print(f"Combined {len(pc_picos)} PICOs for {country} with outcomes: {country_outcomes[:50]}...")
        
        return combined_results

    def extract_picos(
        self,
        source_type: Optional[str] = None,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs using split extraction approach (Population+Comparator separately from Outcomes).
        """
        source_type_to_use = source_type or self.source_type
        
        print(f"Starting PICO extraction for {source_type_to_use}")
        
        print("Step 1: Extracting Population + Comparator")
        pc_results = self.extract_population_comparator(
            source_type=source_type_to_use,
            indication=indication,
            model_override=model_override
        )
        
        print("Step 2: Extracting Outcomes")
        outcomes_results = self.extract_outcomes(
            source_type=source_type_to_use,
            indication=indication,
            model_override=model_override
        )
        
        print("Step 3: Combining results")
        combined_results = self.combine_split_results(pc_results, outcomes_results)
        
        if combined_results:
            indication_for_save = (combined_results[0].get("Indication") if combined_results else 
                                 indication or "unknown indication")
            self.save_extracted_picos(combined_results, source_type_to_use, indication_for_save)
        
        return combined_results

    def save_extracted_picos(
        self,
        extracted_picos: List[Dict[str, Any]],
        source_type: str,
        indication: str
    ):
        """
        Save extracted PICOs to JSON file in the PICO output directory.
        """
        timestamp = datetime.now().isoformat()
        
        organized_data = {
            "extraction_metadata": {
                "timestamp": timestamp,
                "source_type": source_type,
                "indication": indication,
                "total_countries": len(extracted_picos),
                "total_picos": sum(len(country.get("PICOs", [])) for country in extracted_picos)
            },
            "picos_by_country": {}
        }
        
        for country_picos in extracted_picos:
            country = country_picos.get("Country", "Unknown")
            organized_data["picos_by_country"][country] = country_picos
        
        filename = f"{source_type}_picos.json"
        filepath = os.path.join(self.pico_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved organized PICOs to {filepath}")
        
        return filepath