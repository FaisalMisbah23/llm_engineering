# LLM Engineering - Course Projects

This directory contains all 8 completed projects from the LLM Engineering course.

## Project List

### 1. AI-powered Brochure Generator (Week 1)
**Directory:** `1_AI_powered_brochure_generator/`
- **Files:** `brochure_generator.py`, `scraper.py`
- **Description:** Takes a company website URL and generates a professional brochure using AI. Includes web scraping, OpenAI/Ollama integration, and clean output formatting.
- **Key Technologies:** OpenAI Python SDK, BeautifulSoup, dotenv

### 2. Multi-modal Airline Support Agent (Week 2)
**Directory:** `2_Multi_modal_airline_support_agent/`
- **Files:** `airline_assistant.py`
- **Description:** AI customer support assistant for an airline (PIA-AI). Supports multiple LLM providers (Groq, Gemini, Ollama, Cohere) via unified OpenAI-compatible interface. Includes Gradio chat interface.
- **Key Technologies:** Gradio, OpenAI SDK, multi-provider LLM routing

### 3. Synthetic Job Posting Generator (Week 3)
**Directory:** `3_Synthetic_Job_Posting_Generator/`
- **Files:** `job_posting_generator.py`
- **Description:** Generates realistic, validated job postings using LLMs. Includes Pydantic data models for validation, Groq API integration, and structured output generation.
- **Key Technologies:** Pydantic validation, Groq API, structured generation

### 4. Python-to-C++ AI Optimizer (Week 4)
**Directory:** `4_Python_to_Cpp_AI_optimizer/`
- **Files:** `main.cpp`, `system_info.py`
- **Description:** High-performance C++ implementation of maximum subarray sum with LCG random generation. Includes Python system introspection tool for hardware/software profiling.
- **Key Technologies:** C++ optimization, STL, system introspection

### 5. RAG-based AI Knowledge Worker (Week 5)
**Directory:** `5_RAG_based_AI_knowledge_worker/`
- **Files:** `rag_knowledge_worker.py`, `implementation/`, `pro_implementation/`
- **Description:** Advanced RAG system with document ingestion, chunking, vector storage (ChromaDB), and semantic retrieval. Supports both standard and pro implementations.
- **Key Technologies:** ChromaDB, HuggingFace embeddings, LangChain, LiteLLM

### 6. Capstone A: Price Prediction (Frontier Models) (Week 6)
**Directory:** `6_Capstone_A_Price_prediction_Frontier_models/`
- **Files:** `pricer/`
- **Description:** Deep neural network for price prediction using PyTorch. Includes residual blocks, layer normalization, dropout regularization, and GPU training pipeline.
- **Key Technologies:** PyTorch, ResNet-style architecture, NumPy, scikit-learn

### 7. Capstone B: Fine-tuned Open-source Price Prediction (Week 7)
**Directory:** `7_Capstone_B_Fine_tuned_open_source_price_prediction/`
- **Files:** `pricer/`, `util.py`
- **Description:** Open-source price prediction system with evaluation framework. Builds on Week 6 with fine-tuning capabilities.
- **Key Technologies:** PyTorch, model fine-tuning, evaluation metrics

### 8. Capstone C: Autonomous Multi-agent Deal Spotter (Week 8)
**Directory:** `8_Capstone_C_Autonomous_multi_agent_deal_spotter/`
- **Files:** `price_is_right.py`, `deal_agent_framework.py`, `agents/`, `products_vectorstore/`
- **Description:** Sophisticated multi-agent system for deal discovery and evaluation. Features coordinated agents (scanner, ensemble, planning, messaging) with RAG-based product analysis and Gradio dashboard.
- **Key Technologies:** Multi-agent orchestration, ChromaDB, Gradio, RAG pipeline

## Course Completion Summary

All 8 projects completed successfully, covering:
- Week 1: Basic LLM integration and prompt engineering
- Week 2: Multi-provider LLM routing and Gradio interfaces
- Week 3: Structured output generation and validation
- Week 4: Performance optimization (Python to C++)
- Week 5: RAG architecture and knowledge retrieval
- Week 6: Neural networks for prediction (PyTorch)
- Week 7: Model fine-tuning and evaluation
- Week 8: Complex multi-agent orchestration
