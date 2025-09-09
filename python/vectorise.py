import os
import json
import glob
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


import shutil
import numpy as np
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.express as px



import os
import json
import glob
import time
from typing import List
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from collections import defaultdict

class Chunker:
    def __init__(
        self,
        json_folder_path,
        output_dir="./data/text_chunked",
        chunk_size=600,
        maintain_folder_structure=False,
        max_chunk_size=1500,
        min_chunk_size=100
    ):
        self.json_folder_path = json_folder_path
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.maintain_folder_structure = maintain_folder_structure
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.start_time = None
        os.makedirs(self.output_dir, exist_ok=True)

    def start_timer(self):
        self.start_time = time.time()

    def end_timer_and_print(self):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            print(f"Total processing time: {elapsed_time:.2f} seconds")
        else:
            print("Timer was not started.")

    def load_json_documents(self) -> List[Document]:
        documents = []
        json_files = glob.glob(os.path.join(self.json_folder_path, "**/*.json"), recursive=True)
        print(f"Found {len(json_files)} JSON files.")

        for jf in json_files:
            print(f"Processing file: {jf}")
            rel_path = os.path.relpath(jf, self.json_folder_path)
            
            source_type = "unknown"
            path_lower = jf.lower()
            if "hta submission" in path_lower or "hta submissions" in path_lower:
                source_type = "hta_submission"
            elif "clinical guideline" in path_lower or "clinical guidelines" in path_lower:
                source_type = "clinical_guideline"
            
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "doc_id" not in data or "chunks" not in data:
                print(f"Skipping file {jf}: missing 'doc_id' or 'chunks'.")
                continue

            doc_id = data["doc_id"]
            doc_created_date = data.get("created_date", "unknown_year")
            doc_country = data.get("country", data.get("country:", "unknown"))
            doc_source_type = data.get("source_type", source_type)
            chunks = data["chunks"]

            if not isinstance(chunks, list) or len(chunks) == 0:
                print(f"Skipping file {jf}: 'chunks' is empty or not a list.")
                continue

            for c in chunks:
                if "text" not in c:
                    print(f"Skipping a chunk in {jf}: missing 'text'.")
                    continue

                heading_metadata = {
                    "doc_id": doc_id,
                    "heading": c.get("heading", ""),
                    "start_page": c.get("start_page"),
                    "end_page": c.get("end_page"),
                    "created_date": doc_created_date,
                    "country": doc_country,
                    "source_type": doc_source_type,
                    "original_file_path": rel_path
                }

                documents.append(
                    Document(
                        page_content=c["text"],
                        metadata=heading_metadata
                    )
                )

        print(f"Loaded {len(documents)} heading-level documents.")
        return documents

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = SemanticChunker(
            embeddings=HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"),
            breakpoint_threshold_amount=75,
            min_chunk_size=self.chunk_size
        )

        all_splits = []
        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            for i, split_text in enumerate(splits):
                sub_doc = Document(
                    page_content=split_text,
                    metadata=dict(doc.metadata)
                )
                sub_doc.metadata["split_index"] = i
                all_splits.append(sub_doc)

        print(f"Total chunks after semantic splitting: {len(all_splits)}")
        
        all_splits = self.enforce_max_chunk_size(all_splits)
        all_splits = self.merge_small_chunks(all_splits)
        
        print(f"Final chunk count after size adjustments: {len(all_splits)}")
        return all_splits

    def enforce_max_chunk_size(self, chunks: List[Document]) -> List[Document]:
        result = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= self.max_chunk_size:
                result.append(chunk)
            else:
                text = chunk.page_content
                
                words = text.split()
                mid_point = len(words) // 2
                first_half = " ".join(words[:mid_point])
                second_half = " ".join(words[mid_point:])
                
                sub_texts = [first_half, second_half]
                
                for i, sub_text in enumerate(sub_texts):
                    if len(sub_text) > self.max_chunk_size:
                        further_split = []
                        words_sub = sub_text.split()
                        chunk_words = len(words_sub) // 2
                        further_split.append(" ".join(words_sub[:chunk_words]))
                        further_split.append(" ".join(words_sub[chunk_words:]))
                        sub_texts_final = further_split
                    else:
                        sub_texts_final = [sub_text]
                    
                    for j, final_text in enumerate(sub_texts_final):
                        sub_doc = Document(
                            page_content=final_text,
                            metadata=dict(chunk.metadata)
                        )
                        original_split_index = chunk.metadata.get("split_index", 0)
                        sub_doc.metadata["split_index"] = f"{original_split_index}_{i}_{j}"
                        sub_doc.metadata["oversized_split"] = True
                        result.append(sub_doc)
        
        return result

    def merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        if not chunks:
            return chunks
        
        result = []
        skip_next = False
        
        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue
                
            if len(chunk.page_content) >= self.min_chunk_size:
                result.append(chunk)
            else:
                if result and len(result[-1].page_content) + len(chunk.page_content) <= self.max_chunk_size:
                    prev_chunk = result[-1]
                    prev_chunk.page_content = prev_chunk.page_content + " " + chunk.page_content
                    prev_chunk.metadata["merged"] = True
                elif i < len(chunks) - 1:
                    next_chunk = chunks[i + 1]
                    merged_content = chunk.page_content + " " + next_chunk.page_content
                    merged_doc = Document(
                        page_content=merged_content,
                        metadata=dict(chunk.metadata)
                    )
                    merged_doc.metadata["merged"] = True
                    result.append(merged_doc)
                    skip_next = True
                else:
                    result.append(chunk)
        
        return result

    def save_chunks_to_files(self, chunks: List[Document]):
        grouped = defaultdict(list)
        
        for doc in chunks:
            doc_id = doc.metadata.get("doc_id", "unknown_doc")
            original_file_path = doc.metadata.get("original_file_path", "")
            
            key = (doc_id, original_file_path)
            grouped[key].append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        for (doc_id, original_file_path), doc_chunks in grouped.items():
            if self.maintain_folder_structure and original_file_path:
                original_dir = os.path.dirname(original_file_path)
                target_dir = os.path.join(self.output_dir, original_dir)
                os.makedirs(target_dir, exist_ok=True)
                output_filename = f"{doc_id}_chunked.json"
                out_path = os.path.join(target_dir, output_filename)
            else:
                out_path = os.path.join(self.output_dir, f"{doc_id}_chunked.json")
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"doc_id": doc_id, "chunks": doc_chunks}, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(doc_chunks)} chunks to {out_path}")

    def debug_print_chunks(self, docs: List[Document], num_chunks=5):
        for i, doc in enumerate(docs[:num_chunks]):
            print(f"Chunk {i+1}:")
            print(f"Metadata: {doc.metadata}")
            print(f"Content:\n{doc.page_content[:500]}...\n{'-'*40}")

    def run_pipeline(self):
        self.start_timer()
        heading_docs = self.load_json_documents()
        final_chunks = self.chunk_documents(heading_docs)
        self.save_chunks_to_files(final_chunks)
        self.end_timer_and_print()



class Vectoriser:
    """
    Creates or loads a Chroma vectorstore from chunked JSON documents.
    Supports embeddings from OpenAI or BioBERT.
    """

    def __init__(
        self,
        chunked_folder_path: str = "./data/text_chunked",
        embedding_choice: str = "openai",
        db_parent_dir: str = "./data/vectorstore",
        include_metadata_in_text: bool = True,
        metadata_fields_to_include: List[str] = None,
        metadata_format: str = "**{field}:** ",
        incremental_mode: bool = False
    ):
        self.chunked_folder_path = chunked_folder_path
        self.embedding_choice = embedding_choice.lower()
        self.db_parent_dir = db_parent_dir
        self.db_name = self._get_db_path()
        self.include_metadata_in_text = include_metadata_in_text
        self.metadata_fields_to_include = metadata_fields_to_include or ["heading", "source_type"]
        self.metadata_format = metadata_format
        self.incremental_mode = incremental_mode
        self.manifest_path = os.path.join(self.db_parent_dir, f"{os.path.basename(self.db_name)}_manifest.json")

    def _get_db_path(self) -> str:
        store_name = {
            "openai": "open_ai_vectorstore",
            "biobert": "biobert_vectorstore"
        }.get(self.embedding_choice)
        if store_name is None:
            raise ValueError("Unsupported embedding_choice. Use 'openai' or 'biobert'.")
        return os.path.join(self.db_parent_dir, store_name)

    def _enrich_text_with_metadata(self, text: str, metadata: dict) -> str:
        """
        Prepends relevant metadata fields to the text content for better embedding context.
        """
        if not self.include_metadata_in_text:
            return text
        
        enriched_parts = []
        
        for field in self.metadata_fields_to_include:
            value = metadata.get(field, "").strip()
            if value and value.lower() != "unknown":
                formatted_field = self.metadata_format.format(field=field.replace("_", " ").title())
                enriched_parts.append(f"{formatted_field}{value}")
        
        if enriched_parts:
            prefix = " ".join(enriched_parts) + "\n\n"
            return prefix + text
        
        return text

    def _load_manifest(self) -> dict:
        """
        Loads the manifest file that tracks processed documents.
        """
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"processed_files": {}, "total_chunks": 0}

    def _save_manifest(self, manifest: dict):
        """
        Saves the manifest file.
        """
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _get_file_signature(self, file_path: str) -> str:
        """
        Creates a signature for a file based on its modification time and size.
        """
        stat = os.stat(file_path)
        return f"{stat.st_mtime}_{stat.st_size}"

    def load_chunked_documents(self, only_new: bool = False) -> List[Document]:
        """
        Loads chunked JSON files into Document objects.
        Uses recursive glob to find JSON files in any subdirectory of the chunked_folder_path.
        """
        documents = []
        manifest = self._load_manifest() if only_new else {"processed_files": {}}
        processed_files = manifest.get("processed_files", {})
        new_files_count = 0
        
        json_files = glob.glob(os.path.join(self.chunked_folder_path, "**/*.json"), recursive=True)
        print(f"Found {len(json_files)} JSON files in {self.chunked_folder_path} (including subdirectories).")

        for jf in json_files:
            file_signature = self._get_file_signature(jf)
            rel_file_path = os.path.relpath(jf, self.chunked_folder_path)
            
            if only_new and rel_file_path in processed_files:
                if processed_files[rel_file_path] == file_signature:
                    continue
            
            if only_new:
                new_files_count += 1
                
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            rel_path = os.path.relpath(os.path.dirname(jf), self.chunked_folder_path)
            
            for c in data.get("chunks", []):
                text_content = c.get("text", "").strip()
                if not text_content:
                    continue
                
                metadata = c.get("metadata", {})
                
                if rel_path and rel_path != "." and "folder_path" not in metadata:
                    metadata["folder_path"] = rel_path
                
                enriched_text = self._enrich_text_with_metadata(text_content, metadata)
                documents.append(Document(page_content=enriched_text, metadata=metadata))
            
            processed_files[rel_file_path] = file_signature

        if only_new:
            print(f"Found {new_files_count} new or modified files.")
            
        print(f"Loaded {len(documents)} valid document chunks.")
        
        if only_new and documents:
            manifest["processed_files"] = processed_files
            manifest["total_chunks"] = manifest.get("total_chunks", 0) + len(documents)
            self._save_manifest(manifest)
        
        return documents

    def vectorstore_exists(self) -> bool:
        """
        Checks if vectorstore already exists.
        """
        return os.path.exists(self.db_name) and len(os.listdir(self.db_name)) > 0

    def get_embeddings(self):
        """
        Gets the embedding function based on user's choice.
        """
        if self.embedding_choice == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_choice == "biobert":
            return HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        else:
            raise ValueError("Unsupported embedding_choice. Use 'openai' or 'biobert'.")

    def create_vectorstore(self, docs: List[Document]):
        import time
        start_time = time.time()
        
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)

        embeddings = self.get_embeddings()
        
        texts = [doc.page_content for doc in docs]
        metas = [doc.metadata for doc in docs]

        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metas,
            persist_directory=self.db_name
        )
        vectorstore.persist()
        
        for root, dirs, files in os.walk(self.db_name):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)

        manifest = self._load_manifest()
        manifest["total_chunks"] = len(docs)
        all_docs = self.load_chunked_documents(only_new=False)
        json_files = glob.glob(os.path.join(self.chunked_folder_path, "**/*.json"), recursive=True)
        for jf in json_files:
            rel_file_path = os.path.relpath(jf, self.chunked_folder_path)
            manifest["processed_files"][rel_file_path] = self._get_file_signature(jf)
        self._save_manifest(manifest)
        
        elapsed_time = time.time() - start_time
        print(f"Vectorstore created with {len(docs)} chunks at '{self.db_name}'. Time taken: {elapsed_time:.2f} seconds.")
        
        return vectorstore

    def add_new_documents(self, docs: List[Document]):
        """
        Adds new documents to an existing vectorstore.
        """
        if not docs:
            print("No new documents to add.")
            return
            
        if not self.vectorstore_exists():
            raise ValueError("Vectorstore does not exist. Create it first using create_vectorstore().")
        
        import time
        start_time = time.time()
        
        embeddings = self.get_embeddings()
        vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=embeddings
        )
        
        texts = [doc.page_content for doc in docs]
        metas = [doc.metadata for doc in docs]
        
        vectorstore.add_texts(texts=texts, metadatas=metas)
        vectorstore.persist()
        
        elapsed_time = time.time() - start_time
        print(f"Added {len(docs)} new chunks to existing vectorstore at '{self.db_name}'. Time taken: {elapsed_time:.2f} seconds.")
        return vectorstore

    def load_vectorstore(self):
        """
        Loads an existing Chroma vectorstore from disk.
        """
        embeddings = self.get_embeddings()
        vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=embeddings
        )
        print(f"Vectorstore loaded from '{self.db_name}'.")
        return vectorstore

    def run_pipeline(self):
        """
        Main pipeline method to either load or create the vectorstore.
        """
        if self.incremental_mode and self.vectorstore_exists():
            print("Running in incremental mode. Checking for new documents...")
            new_docs = self.load_chunked_documents(only_new=True)
            
            if new_docs:
                print(f"Found {len(new_docs)} new document chunks. Adding to existing vectorstore...")
                return self.add_new_documents(new_docs)
            else:
                print("No new documents found. Loading existing vectorstore...")
                return self.load_vectorstore()
        
        elif self.vectorstore_exists() and not self.incremental_mode:
            print("Vectorstore exists. Loading now...")
            return self.load_vectorstore()
        
        else:
            print("No vectorstore found or incremental mode disabled. Creating new one...")
            docs = self.load_chunked_documents(only_new=False)
            if not docs:
                raise ValueError("No valid documents found for vectorization.")
            return self.create_vectorstore(docs)

    def visualize_vectorstore(self, vectorstore):
        """
        Visualizes the vectorstore using t-SNE, grouped by doc_id.
        """
        result = vectorstore.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']

        doc_ids = [
            md.get('doc_id', 'unknown') if isinstance(md, dict) else 'unknown'
            for md in metadatas
        ]

        unique_docs = sorted(set(doc_ids))
        colors_palette = px.colors.qualitative.Safe
        color_dict = {
            doc: colors_palette[i % len(colors_palette)]
            for i, doc in enumerate(unique_docs)
        }
        colors = [color_dict[doc] for doc in doc_ids]

        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        fig = go.Figure()

        for doc in unique_docs:
            indices = [i for i, d in enumerate(doc_ids) if d == doc]
            fig.add_trace(go.Scatter(
                x=reduced_vectors[indices, 0],
                y=reduced_vectors[indices, 1],
                mode='markers',
                name=doc,
                marker=dict(size=6, opacity=0.8, color=color_dict[doc]),
                text=[f"Doc ID: {doc}<br>{documents[i][:150]}..." for i in indices],
                hoverinfo='text'
            ))

        fig.update_layout(
            title='2D Vectorstore Visualization (t-SNE) grouped by Document',
            legend_title="Document ID",
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        fig.show()