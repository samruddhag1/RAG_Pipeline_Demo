# RAG Design Report

## Chunking Strategy
- Size: 5000 chars + 50 char overlap
- Preserves section metadata (Item X, page #)

## Retrieval
- Hybrid: 70% semantic (all-MiniLM-L6-v2) + 30% BM25
- Top-5 chunks

## LLM Choice
- Mistral 7B Instruct Q4_K_M (4-bit quantized)
- Local inference via llama.cpp

## Out-of-Scope Handling
- Strict prompt: "ONLY provided context"
- Keyword detection for refusal phrases
