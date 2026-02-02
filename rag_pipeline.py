import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path

# PDF parsing
import pypdf

# Embeddings & Vector DB
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Reranking algorithm module
from rank_bm25 import BM25Okapi

# LLM (local inference)
from llama_cpp import Llama

class DocumentProcessor:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(self.model_name)

    def parse_pdf(self, pdf_path: str, doc_name: str) -> List[Dict]:
        """Parse PDF and create chunks with metadata about sources"""
        chunks = []
        reader = pypdf.PdfReader(pdf_path)

        prev_section = 'Unknown'
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            # Clean text and split into chunks
            cleaned_text = re.sub(r'\s+', ' ', text.strip())

            # Split into 4000-char chunks with 50-char overlap
            chunk_size = 4000
            overlap = 50
            start = 0
            while start < len(cleaned_text):
                end = min(start + chunk_size, len(cleaned_text))
                chunk_text = cleaned_text[start:end]

                # Extract section headers (look for "Item X")
                section = self.extract_section(chunk_text)
                if not section:
                  section = prev_section
                prev_section = section

                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'doc_name': doc_name,
                        'section': section,
                        'page': page_num + 1,
                        'chunk_id': len(chunks)
                    }
                })
                start += chunk_size - overlap

        return chunks

    def extract_section(self, text: str) -> str:
        """Extract section like 'Item 1B', 'Item 8'"""
        section_match = re.search(r'Item\s+(\d+[A-Z]?)', text, re.IGNORECASE)
        return section_match.group(0) if section_match else None

class VectorStore:
    def __init__(self, chunks: List[Dict], processor):
        self.chunks = chunks
        self.processor = processor
        self.texts = [chunk['content'] for chunk in chunks]
        self.embeddings = self.processor.embedding_model.encode(self.texts)

        # FAISS index
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))

        # BM25 for re-ranking
        tokenized_texts = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Semantic and  BM25 reranking function"""
        # Semantic search (top 10)
        query_emb = self.processor.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb.astype('float32'), 10)

        # BM25 scores for reranking
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Combine scores
        candidates = []
        for idx in indices[0]:
            semantic_score = distances[0][list(indices[0]).index(idx)]
            hybrid_score = 0.7 * semantic_score + 0.3 * bm25_scores[idx]
            candidates.append((idx, hybrid_score))

        # Top-K by score
        top_k = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
        return [self.chunks[i] for i, _ in top_k]

class RAGSystem:
    def __init__(self):
        self.llm = Llama(
            model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=5120,
            n_threads=4,
            n_gpu_layers=16  # Use GPU if available
        )

    def format_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Strict prompt with citation requirements"""
        context = "\n\n".join([
            f"[{chunk['metadata']['doc_name']}, {chunk['metadata']['section']}, p. {chunk['metadata']['page']}]:\n{chunk['content']}"
            for chunk in context_chunks
        ])

        prompt = f"""You are a financial analyst. The context provided to you is part of 
real SEC filings containing financial statements, risk factors, legal proceedings,
and executive compensation details. Answer using ONLY the provided context.
Cite sources exactly as shown in brackets. Answer only based on the facts in the context.
Do not answer anything out of scope of context. If not in context, say: "Not specified in the document." In this case sources should be an empty list.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with citations):"""
        return prompt

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            stop=["CONTEXT:", "QUESTION:", "\n\n"]
        )
        return output['choices'][0]['text'].strip()
      

