# ⚡ Quick Start Guide

Get the Insurellm RAG Assistant running in under 5 minutes!

## 🎯 Prerequisites Checklist

- [ ] Python 3.12 or higher installed
- [ ] OpenAI API key (get one at [platform.openai.com](https://platform.openai.com))
- [ ] 5GB free disk space
- [ ] Internet connection for installing dependencies

## 🚀 Installation (2 minutes)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd insurellm-rag-assistant

# Create and activate virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- LangChain (RAG framework)
- OpenAI (LLM and embeddings)
- Chroma (vector database)
- Gradio (web UI)
- And other required packages

### Step 3: Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

Add this line to `.env`:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

## 🎮 Running the Basic RAG System (1 minute)

### Step 1: Create the Vector Database

```bash
python implementation/ingest.py
```

You should see:
```
Loaded 76 documents
There are 413 vectors with 3,072 dimensions in the vector store
Ingestion complete
```

This creates a `vector_db/` directory with your knowledge base.

### Step 2: Launch the Chat Interface

```bash
python app.py
```

Your browser will automatically open to `http://localhost:7860`

## 💬 Try These Example Questions

Once the chat interface loads, try asking:

### Simple Facts
- "Who is the CEO of Insurellm?"
- "What products does Insurellm offer?"
- "How many employees does Insurellm have?"

### Employee Queries
- "Who won the IIOTY award in 2023?"
- "What is James Wilson's salary?"
- "Tell me about Priya Sharma's career at Insurellm"

### Product Questions
- "What is Carllm and how much does it cost?"
- "What features does Healthllm offer?"
- "Compare the pricing of Homellm tiers"

### Contract Details
- "What is the DriveSmart Insurance contract worth?"
- "Who are Insurellm's biggest clients?"
- "What is the duration of the Metropolitan Life Group contract?"

## 🔧 Troubleshooting

### Issue: "OpenAI API Key not set"
**Solution**: Make sure your `.env` file has `OPENAI_API_KEY=sk-...` and you've restarted the app.

### Issue: "No module named 'langchain'"
**Solution**: Activate your virtual environment and run `pip install -r requirements.txt` again.

### Issue: "Port 7860 already in use"
**Solution**: Either close the other application or edit `app.py` to use a different port:
```python
ui.launch(inbrowser=True, server_port=7861)
```

### Issue: Slow responses
**Solution**: The first query creates the embeddings cache. Subsequent queries will be faster. If still slow, consider using `gpt-4o-mini` instead of `gpt-4.1-nano` in `implementation/answer.py`.

## 🎓 Next Steps

### Option 1: Try the Advanced RAG System

```bash
# Preprocess the knowledge base (takes 3-5 minutes)
python pro_implementation/ingest.py

# Launch the advanced system
python pro_app.py
```

The advanced system uses:
- LLM-powered document chunking
- Query rewriting
- Result reranking
- Better accuracy!

### Option 2: Run Evaluations

```bash
# Evaluate a specific test question
python evaluation/eval.py 0

# Launch the evaluation dashboard
python evaluator.py
```

The dashboard shows:
- Retrieval quality metrics (MRR, nDCG)
- Answer quality scores (accuracy, completeness, relevance)
- Performance by question category

### Option 3: Explore the Notebooks

Open the Jupyter notebooks to understand how everything works:

```bash
jupyter notebook
```

Then explore:
1. `day1.ipynb` - Naive keyword matching
2. `day2.ipynb` - Vector embeddings and visualization
3. `day3.ipynb` - Building RAG with LangChain
4. `day4.ipynb` - Evaluation framework
5. `day5.ipynb` - Advanced techniques

## 📊 Understanding the Interface

### Chat Interface Components

```
┌─────────────────────────────────────────────────────┐
│  💬 Conversation          │  📚 Retrieved Context  │
│                            │                         │
│  Your Question            │  Source: employees/...  │
│  ↓                        │  [Relevant chunk text]  │
│  Assistant's Answer       │                         │
│                            │  Source: products/...   │
│  [Your next question]     │  [Relevant chunk text]  │
└─────────────────────────────────────────────────────┘
```

- **Left side**: Chat conversation
- **Right side**: Retrieved context chunks that informed the answer
- **Copy button**: Copy any response to clipboard

## ⚙️ Customization Quick Tips

### Change the LLM Model

Edit `implementation/answer.py`:
```python
MODEL = "gpt-4o-mini"  # Faster and cheaper
# or
MODEL = "gpt-4-turbo"  # More capable
```

### Adjust Retrieval Count

Edit `implementation/answer.py`:
```python
RETRIEVAL_K = 5   # Fewer chunks (faster, less context)
RETRIEVAL_K = 15  # More chunks (slower, more context)
```

### Use HuggingFace Embeddings (Free!)

Edit `implementation/answer.py`:
```python
# Comment out:
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Uncomment:
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Then re-run `python implementation/ingest.py` to recreate the database.

## 🎯 Performance Expectations

### Basic RAG System
- **Ingestion**: ~30 seconds
- **First query**: ~3-5 seconds (includes model loading)
- **Subsequent queries**: ~1-2 seconds
- **Accuracy**: ~85% on test suite

### Advanced RAG System
- **Ingestion**: ~3-5 minutes (LLM preprocessing)
- **First query**: ~5-8 seconds (includes reranking)
- **Subsequent queries**: ~2-3 seconds
- **Accuracy**: ~92% on test suite

## 📚 Recommended Learning Path

1. **Day 1** (15 min): Run the basic system, ask questions
2. **Day 2** (30 min): Explore `day1.ipynb` and `day2.ipynb`
3. **Day 3** (30 min): Study `day3.ipynb` and understand the code
4. **Day 4** (45 min): Run evaluations, review `day4.ipynb`
5. **Day 5** (1 hour): Try advanced system, study `day5.ipynb`

## 🆘 Getting Help

1. **Error messages**: Read carefully - they usually tell you exactly what's wrong
2. **Check the notebooks**: They contain detailed explanations
3. **Review the evaluation results**: See how the system performs on different question types
4. **Experiment**: Try modifying parameters and see what changes

## ✅ Success Checklist

You've successfully set up the system when you can:

- [ ] Ask "Who is the CEO?" and get "Avery Lancaster"
- [ ] See retrieved context displayed on the right side
- [ ] Ask follow-up questions in the conversation
- [ ] See relevant documents being retrieved
- [ ] Get accurate answers about employees, products, and contracts

---

**🎉 Congratulations!** You now have a working RAG system. Start exploring and learning!

For more details, see the main [README.md](README.md)
