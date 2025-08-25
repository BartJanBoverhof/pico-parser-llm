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
    Enhanced PICO extractor that works with pre-stored chunks from the retrieval step.
    """
    def __init__(
        self,
        system_prompt: str,
        user_prompt_template: str,
        model_name: str = "gpt-4o-mini",
        results_output_dir: str = "results",
        max_tokens: int = 12000
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.results_output_dir = results_output_dir
        self.chunks_input_dir = os.path.join(results_output_dir, "chunks")
        self.pico_output_dir = os.path.join(results_output_dir, "PICO")
        
        os.makedirs(self.pico_output_dir, exist_ok=True)
        
        self.openai = OpenAI()
        
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the current encoding."""
        return len(self.encoding.encode(text))

    def find_chunk_file(self, source_type: str, indication: Optional[str] = None) -> Optional[str]:
        """
        Find the appropriate chunk file based on source_type and indication.
        """
        if not os.path.exists(self.chunks_input_dir):
            print(f"Chunks directory not found: {self.chunks_input_dir}")
            return None
        
        files = os.listdir(self.chunks_input_dir)
        chunk_files = [f for f in files if f.endswith('_retrieval_results.json')]
        
        if indication:
            indication_short = indication.split()[0].lower() if indication else "unknown"
            target_filename = f"{source_type}_{indication_short}_retrieval_results.json"
            if target_filename in chunk_files:
                return os.path.join(self.chunks_input_dir, target_filename)
        
        source_filename = f"{source_type}_retrieval_results.json"
        if source_filename in chunk_files:
            return os.path.join(self.chunks_input_dir, source_filename)
        
        for filename in chunk_files:
            if source_type in filename:
                return os.path.join(self.chunks_input_dir, filename)
        
        print(f"No chunk file found for source_type: {source_type}, indication: {indication}")
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

    def extract_picos_from_context(
        self,
        context: str,
        indication: str,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> Dict[str, Any]:
        """
        Extract PICOs from context using OpenAI API.
        """
        if not context.strip():
            return {
                "Indication": indication,
                "Country": None,
                "PICOs": []
            }
        
        try:
            user_prompt = self.user_prompt_template.format(
                indication=indication,
                context_block=context
            )
            
            if model_override and isinstance(model_override, ChatOpenAI):
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = model_override.invoke(messages)
                result_text = response.content
            else:
                model_to_use = model_override if isinstance(model_override, str) else self.model_name
                
                response = self.openai.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
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
                print(f"Failed to parse JSON response. Raw response: {result_text[:500]}...")
                return {
                    "Indication": indication,
                    "Country": None,
                    "PICOs": [],
                    "Error": "JSON parsing failed",
                    "RawResponse": result_text
                }
                
        except Exception as e:
            print(f"Error in PICO extraction: {e}")
            return {
                "Indication": indication,
                "Country": None,
                "PICOs": [],
                "Error": str(e)
            }

    def extract_picos(
        self,
        source_type: str,
        indication: Optional[str] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs from pre-stored chunks for a specific source type.
        """
        chunk_file_path = self.find_chunk_file(source_type, indication)
        if not chunk_file_path:
            print(f"No chunk file found for source_type: {source_type}")
            return []
        
        print(f"Loading chunks from: {chunk_file_path}")
        chunk_data = self.load_chunks_from_file(chunk_file_path)
        
        if not chunk_data:
            print("No chunk data loaded")
            return []
        
        results_by_country = chunk_data.get("results_by_country", {})
        indication_from_metadata = chunk_data.get("retrieval_metadata", {}).get("indication") or indication or "unknown indication"
        
        extracted_picos = []
        
        for country, country_data in results_by_country.items():
            chunks = country_data.get("chunks", [])
            
            if not chunks:
                print(f"No chunks found for country: {country}")
                continue
            
            print(f"Processing {len(chunks)} chunks for {country}")
            
            context = self.build_context_from_chunks(chunks)
            if not context:
                print(f"No context built for country: {country}")
                continue
            
            pico_result = self.extract_picos_from_context(
                context=context,
                indication=indication_from_metadata,
                model_override=model_override
            )
            
            pico_result["Country"] = country
            pico_result["ChunksUsed"] = len(chunks)
            pico_result["ContextTokens"] = self.count_tokens(context)
            
            extracted_picos.append(pico_result)
            
            print(f"Extracted {len(pico_result.get('PICOs', []))} PICOs for {country}")
        
        if extracted_picos:
            self.save_extracted_picos(extracted_picos, source_type, indication_from_metadata)
        
        return extracted_picos

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
        
        if indication:
            indication_short = indication.split()[0].lower() if indication else "unknown"
            filename = f"{source_type}_{indication_short}_picos_organized.json"
        else:
            filename = f"{source_type}_picos_organized.json"
        
        filepath = os.path.join(self.pico_output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(organized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved organized PICOs to {filepath}")
        
        for country_picos in extracted_picos:
            country = country_picos.get("Country", "Unknown")
            individual_filename = f"{source_type}_picos_{country}.json"
            individual_filepath = os.path.join(self.pico_output_dir, individual_filename)
            
            with open(individual_filepath, "w", encoding="utf-8") as f:
                json.dump(country_picos, f, indent=2, ensure_ascii=False)
        
        return filepath