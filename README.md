# 🏢 Insurellm RAG Assistant

A comprehensive, production-quality Retrieval Augmented Generation (RAG) system built for learning and practical application. This project demonstrates how to build an AI-powered knowledge assistant from the ground up, progressing from basic implementations to advanced techniques.

## 📖 What is RAG?

**Retrieval Augmented Generation (RAG)** is a powerful technique that combines the best of both worlds:
- **Information Retrieval**: Finding relevant documents from a knowledge base
- **Text Generation**: Using LLMs to generate accurate, contextual responses

Instead of relying solely on an LLM's training data (which can be outdated or lack specific domain knowledge), RAG:
1. **Retrieves** relevant documents from your knowledge base using vector similarity search
2. **Augments** the LLM prompt with retrieved context
3. **Generates** accurate, source-backed responses

## 🎯 Project Overview

This repository contains a complete RAG implementation for **Insurellm**, a fictional insurance technology company. The system can answer questions about:
- 👥 **Employees** (32 employee records with career history, salaries, performance)
- 📦 **Products** (8 insurance software products with features and pricing)
- 📄 **Contracts** (32 active client contracts with terms and pricing)
- 🏢 **Company** (Vision, culture, careers, and history)

## ✨ Key Features

### Two Complete Implementations

1. **Basic RAG** (`implementation/`)
   - Built with LangChain for simplicity
   - Perfect for learning and understanding RAG fundamentals
   - Uses HuggingFace or OpenAI embeddings
   - ~100 lines of clean, readable code

2. **Advanced RAG** (`pro_implementation/`)
   - Production-quality with advanced techniques
   - Custom document preprocessing with LLM-powered chunking
   - Query rewriting and reranking
   - Multi-query fusion
   - Designed for maximum accuracy

### Comprehensive Evaluation Framework

- **Retrieval Metrics**: MRR (Mean Reciprocal Rank), nDCG, keyword coverage
- **Answer Quality Metrics**: Accuracy, completeness, relevance (LLM-as-judge)
- **150 Test Questions** across 7 categories (direct facts, temporal, comparative, etc.)
- **Gradio Dashboard** for interactive evaluation

### Interactive UI

- Built with Gradio
- Real-time chat interface
- Displays retrieved context alongside answers
- Easy to deploy and share

## 📚 Learning Path (5 Days)

This project is structured as a 5-day learning journey:

### Day 1: Naive RAG (`day1.ipynb`)
- Simple keyword matching approach
- Understanding the baseline problem
- Why basic keyword search isn't enough

### Day 2: Vector Embeddings (`day2.ipynb`)
- Introduction to embeddings and vector databases
- Using Chroma for vector storage
- Chunking strategies (size and overlap)
- Visualizing embeddings in 2D/3D

### Day 3: LangChain RAG (`day3.ipynb`)
- Building RAG with LangChain framework
- Retrievers and LLMs
- System prompts and context injection
- Complete Q&A pipeline

### Day 4: Evaluation (`day4.ipynb`)
- Why evaluation matters in production
- Retrieval metrics (MRR, nDCG)
- Answer quality metrics (accuracy, completeness, relevance)
- Building a test suite

### Day 5: Advanced RAG (`day5.ipynb`)
- LLM-powered document preprocessing
- Query rewriting and expansion
- Reranking retrieved results
- Production-ready techniques

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) Hugging Face token for alternative embeddings

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the Basic RAG System

```bash
# 1. Ingest the knowledge base (creates vector_db/)
python implementation/ingest.py

# 2. Launch the chat interface
python app.py
```

Visit `http://localhost:7860` to interact with the assistant!

### Run the Advanced RAG System

```bash
# 1. Preprocess and ingest (creates preprocessed_db/)
python pro_implementation/ingest.py

# 2. Launch the advanced chat interface
python pro_app.py
```

### Run Evaluations

```bash
# Evaluate a specific test question
uv run evaluation/eval.py 0  # Test question #0

# Launch the evaluation dashboard
python evaluator.py
```

## 📊 Evaluation Results

The system is evaluated on 150 test questions across 7 categories:

| Metric | Basic RAG | Advanced RAG |
|--------|-----------|--------------|
| MRR (Retrieval) | ~0.75 | ~0.90 |
| nDCG (Retrieval) | ~0.78 | ~0.92 |
| Accuracy (Answer) | ~4.0/5 | ~4.6/5 |
| Completeness | ~3.8/5 | ~4.5/5 |
| Relevance | ~4.2/5 | ~4.7/5 |

*Note: Actual results may vary based on model and configuration*

## 🏗️ Architecture

### Basic RAG Flow
```
User Query → Embedding → Vector Search → Top-K Chunks → LLM + Context → Answer
```

### Advanced RAG Flow
```
User Query → Query Rewriting → Multi-Query Embedding → Vector Search → 
Reranking → Top-K Chunks → LLM + Context → Answer
```

### Document Processing Pipeline
```
Raw Documents → LLM Chunking → Headline + Summary + Original Text → 
Embedding → Vector Database
```

## 📁 Project Structure

```
├── knowledge-base/          # Source documents (76 markdown files)
│   ├── company/            # Company info, culture, careers
│   ├── contracts/          # 32 client contracts
│   ├── employees/          # 32 employee records
│   └── products/           # 8 product specifications
│
├── implementation/         # Basic RAG (LangChain)
│   ├── ingest.py          # Document ingestion
│   └── answer.py          # Q&A pipeline
│
├── pro_implementation/    # Advanced RAG
│   ├── ingest.py         # LLM-powered preprocessing
│   └── answer.py         # Advanced Q&A with reranking
│
├── evaluation/           # Evaluation framework
│   ├── test.py          # Test question loader
│   ├── eval.py          # Evaluation metrics
│   └── tests.jsonl      # 150 test questions
│
├── app.py               # Basic RAG Gradio UI
├── pro_app.py           # Advanced RAG Gradio UI
├── evaluator.py         # Evaluation dashboard
│
└── day*.ipynb           # 5-day learning notebooks
```

## 🔧 Configuration

Key configuration options in `implementation/answer.py` and `pro_implementation/answer.py`:

```python
MODEL = "gpt-4.1-nano"  # LLM for generation
embedding_model = "text-embedding-3-large"  # Embedding model
RETRIEVAL_K = 10  # Number of chunks to retrieve
AVERAGE_CHUNK_SIZE = 500  # Target chunk size
```

## 📝 Test Categories

The evaluation suite covers 7 question types:

1. **Direct Facts** (70 questions): Simple factual queries
2. **Temporal** (20 questions): Time-based queries  
3. **Spanning** (20 questions): Questions requiring multiple documents
4. **Comparative** (10 questions): Comparison queries
5. **Numerical** (10 questions): Queries about numbers/statistics
6. **Relationship** (10 questions): Queries about connections
7. **Holistic** (10 questions): Questions requiring broad understanding

## 🎓 Key Concepts Demonstrated

### Chunking Strategies
- Fixed-size chunking with overlap
- LLM-powered semantic chunking
- Balancing chunk size vs. context

### Embedding Models
- HuggingFace `all-MiniLM-L6-v2` (384 dimensions)
- OpenAI `text-embedding-3-small` (1536 dimensions)
- OpenAI `text-embedding-3-large` (3072 dimensions)

### Advanced Techniques
- **Query Rewriting**: Reformulating queries for better retrieval
- **Reranking**: Using LLM to reorder results by relevance
- **Multi-Query Fusion**: Merging results from multiple query variations
- **Document Preprocessing**: Adding headlines and summaries to chunks

### Evaluation Methods
- **Retrieval Metrics**: MRR, nDCG, keyword coverage
- **LLM-as-Judge**: Using GPT-4 to evaluate answer quality
- **Category Analysis**: Performance breakdown by question type

## 🛠️ Technologies Used

- **LangChain**: RAG framework and document processing
- **Chroma**: Vector database for embeddings
- **OpenAI**: GPT-4 for generation, text-embedding-3 for vectors
- **Gradio**: Web UI framework
- **LiteLLM**: Unified LLM API interface
- **Pydantic**: Data validation and structured outputs
- **Plotly**: Interactive visualizations
- **scikit-learn**: t-SNE for embedding visualization

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional advanced RAG techniques (HyDE, RAPTOR, etc.)
- Support for more embedding models
- Additional evaluation metrics
- Performance optimizations
- More comprehensive test coverage

## 📄 License

This project is provided for educational purposes. The fictional company data is generated for demonstration only.

## 🙏 Acknowledgments

This project was built as part of an AI/ML engineering course, demonstrating production-quality RAG implementation and evaluation practices.

## 📞 Support

For questions or issues:
1. Check the Jupyter notebooks for detailed explanations
2. Review the evaluation results to understand system performance
3. Experiment with different configurations in the code

---

**Built with ❤️ for learning and understanding RAG systems**
