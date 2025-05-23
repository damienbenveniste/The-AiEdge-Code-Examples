# The AI Edge - Code Examples

A comprehensive collection of educational code examples covering modern AI and machine learning topics, from fundamental transformer architectures to advanced LLM applications and deployment strategies.

## 📁 Project Structure

### 🔤 LLMs (Large Language Models)
Examples covering the fundamentals of LLMs, from basic architectures to advanced training and deployment.

#### Basic Transformer Architecture
- **Location**: `code-examples/LLMs/Basic-Transformer-Architecture/`
- **Description**: Complete implementation of transformer architecture from scratch
- **Files**:
  - `components.py` - Core transformer components (attention, positional encoding, feed-forward)
  - `transformer.py` - Complete encoder-decoder transformer implementation
  - `test_code.py` - Example usage and testing code

#### Training LLMs
- **Location**: `code-examples/LLMs/Training-LLMS/`
- **Description**: Jupyter notebook demonstrating LLM training techniques
- **Files**: `training.ipynb`

#### Fine-tuning with LoRA/QLoRA
- **Location**: `code-examples/LLMs/Finetuning-LoRA-QLoRA/`
- **Description**: Parameter-efficient fine-tuning techniques
- **Files**: `finetuning.ipynb`

#### Distributed Training
- **Location**: `code-examples/LLMs/Distributed_training/`
- **Description**: Multi-GPU and distributed training setup
- **Files**:
  - `src/model.py` - Model definitions
  - `src/training.py` - Training loops
  - `src/data_processing.py` - Data handling utilities
  - `src/requirements.txt` - Dependencies

#### Deployment with vLLM
- **Location**: `code-examples/LLMs/Deploying-with-vLLM-basic/`
- **Description**: Basic LLM deployment using vLLM
- **Files**:
  - `client.py` - Client code for API interaction
  - `deploy.sh` - Deployment script

### 🤖 LLM Applications
Real-world applications and advanced patterns using LLMs.

#### Basic RAG (Retrieval-Augmented Generation)
- **Location**: `code-examples/LLM-Applications/Basic-RAG/`
- **Description**: Fundamental RAG implementation
- **Files**:
  - `api.py` - REST API for RAG system
  - `indexing_pipeline.py` - Document indexing pipeline
  - `indexing/` - Data loading, transformation, and indexing modules
  - `retrieval/` - Conversation QA implementation
  - `test_code.ipynb` - Interactive testing notebook

#### GraphRAG
- **Location**: `code-examples/LLM-Applications/GraphRAG/`
- **Description**: Graph-based RAG implementation
- **Files**:
  - `data_models.py` - Graph data structures
  - `generation.py` - Text generation with graph context
  - `indexing/` - Graph extraction and community detection
  - `indexing_pipeline.py` - Complete indexing workflow
  - `book/pride-and-prejudice.txt` - Sample document

#### RAG with LangGraph
- **Location**: `code-examples/LLM-Applications/RAG-with-LangGraph/`
- **Description**: RAG implementation using LangGraph framework
- **Files**:
  - `graph.py` - LangGraph workflow definition
  - `nodes.py` - Individual processing nodes
  - `chains.py` - LLM chains and prompts
  - `websearch.py` - Web search integration
  - `langgraph-rag.ipynb` - Complete tutorial notebook

#### LATS (Language Agent Tree Search)
Two implementations of the LATS algorithm:

##### LATS with Burr
- **Location**: `code-examples/LLM-Applications/LATS-with-Burr/`

##### LATS with LangGraph
- **Location**: `code-examples/LLM-Applications/LATS-with-LangGraph/`

**Common files for both implementations**:
- `graph.py` - State graph implementation
- `nodes.py` - Action and reflection nodes
- `chains.py` - LLM reasoning chains
- `tools.py` - External tool integrations
- `tree.py` - Tree search logic
- `reflection.py` - Self-reflection mechanisms
- `test_code.ipynb` - Testing and examples

#### Complete RAG Application
- **Location**: `code-examples/LLM-Applications/RAG_application/`
- **Description**: Full-stack RAG application with frontend and backend

##### Backend
- **Location**: `code-examples/LLM-Applications/RAG_application/Backend/`
- **Technologies**: FastAPI, SQLAlchemy, Pydantic
- **Files**:
  - `app/main.py` - FastAPI application entry point
  - `app/models.py` - Database models
  - `app/schemas.py` - Pydantic data validation
  - `app/crud.py` - Database operations
  - `app/chains.py` - LLM processing chains
  - `app/data_indexing.py` - Document indexing
  - `Dockerfile` - Container configuration

##### Frontend
- **Location**: `code-examples/LLM-Applications/RAG_application/Frontend/`
- **Technologies**: Streamlit
- **Files**:
  - `app.py` - Main Streamlit application
  - `pages/` - Multi-page application structure

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Additional dependencies vary by example (see individual requirements.txt files)

### Basic Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd The-AiEdge-Code-Examples
   ```

2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies for specific examples:
   ```bash
   cd code-examples/LLMs/Basic-Transformer-Architecture
   pip install torch torchvision
   ```

### Running Examples

#### Basic Transformer
```bash
cd code-examples/LLMs/Basic-Transformer-Architecture
python test_code.py
```

#### RAG Application Backend
```bash
cd code-examples/LLM-Applications/RAG_application/Backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### RAG Application Frontend
```bash
cd code-examples/LLM-Applications/RAG_application/Frontend
pip install -r requirements.txt
streamlit run app.py
```

## 📚 Learning Path

### Beginner
1. **Basic Transformer Architecture** - Understand the fundamentals
2. **Basic RAG** - Learn retrieval-augmented generation
3. **Complete RAG Application** - See end-to-end implementation

### Intermediate
1. **Training LLMs** - Learn training techniques
2. **Fine-tuning with LoRA/QLoRA** - Parameter-efficient methods
3. **RAG with LangGraph** - Advanced workflow orchestration

### Advanced
1. **GraphRAG** - Graph-based knowledge representation
2. **LATS** - Tree search for reasoning
3. **Distributed Training** - Scale training across multiple GPUs
4. **vLLM Deployment** - Production deployment strategies

## 🛠️ Key Technologies

- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face transformers library
- **LangChain/LangGraph** - LLM application frameworks
- **FastAPI** - Modern web API framework
- **Streamlit** - Interactive web applications
- **SQLAlchemy** - Database ORM
- **vLLM** - High-performance LLM serving

## 📖 Educational Focus

This repository is designed for educational purposes, providing:

- **Clear, commented code** with detailed explanations
- **Progressive complexity** from basic to advanced concepts
- **Multiple implementation approaches** for the same concepts
- **Real-world applications** showing practical usage
- **Interactive notebooks** for hands-on learning

## 🤝 Contributing

This is an educational repository. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new examples following the existing structure
- Improve documentation and comments

## 📄 License

This project is for educational purposes. Please check individual dependencies for their respective licenses.

## 🔗 Related Resources

- [The AI Edge Blog](https://www.theaiedge.io)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LangChain Documentation](https://python.langchain.com/)