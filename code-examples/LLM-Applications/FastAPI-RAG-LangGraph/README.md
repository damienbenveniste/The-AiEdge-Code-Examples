# AI Assistant

A sophisticated Retrieval-Augmented Generation (RAG) chat system built with FastAPI, LangGraph, and Streamlit. This application demonstrates advanced AI agent architecture with multi-step reasoning, document retrieval, response evaluation, and iterative improvement.

## Architecture Overview

The system consists of three main components:

### 1. **Multi-Step Chat Agent** (LangGraph)
- **Intent Router**: Determines if queries need knowledge base retrieval
- **Document Retriever**: Two-stage filtering for relevant content
- **Response Generator**: Creates contextual responses with citations
- **Quality Evaluator**: Validates factual accuracy and completeness
- **Fallback Handler**: Graceful error handling and user guidance

### 2. **Knowledge Base Pipeline**
- **Document Parsing**: Processes website content (currently mock CSV data)
- **AI Summarization**: Generates concise summaries using GPT models
- **Vector Storage**: ChromaDB for semantic search and retrieval
- **API Management**: RESTful endpoints for content indexing

### 3. **User Interface**
- **Chat Interface**: Interactive conversation with the AI assistant
- **Admin Interface**: Knowledge base management and content indexing
- **Real-time Feedback**: Progress indicators and status updates

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-application
   ```

2. **Set up the virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Configure environment variables**
   ```bash
   # Create .env file in the backend directory
   echo "OPENAI_API_KEY=your_openai_api_key_here" > backend/.env
   ```

## Running the Application

### Step 1: Start the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API server will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Chat Endpoint**: `POST /chat/message`
- **Indexing Endpoint**: `POST /indexing/index`

### Step 2: Launch the Frontend Interface

```bash
cd frontend
streamlit run main.py
```

The web interface will open at `http://localhost:8501`

## Using the Application

### Chat Interface
1. Navigate to the **Chat** tab
2. Set your User ID in the sidebar (default: 1)
3. Start conversing with the AI assistant
4. The system automatically determines if knowledge retrieval is needed

### Knowledge Base Management
1. Navigate to the **Indexing** tab
2. Enter a URL and user identifier
3. Click "Index Content" to process and store documents
4. New content becomes available for chat queries

## Testing the RAG Pipeline

### Test Knowledge Retrieval
1. First, index some content using the indexing interface
2. Ask specific questions about the indexed content
3. Observe how the AI provides cited responses with source URLs
4. Try general knowledge questions to see direct responses

### Example Interactions
```
User: "What is Python?"
AI: Direct response using pre-trained knowledge

User: "What are the pricing tiers?"  
AI: Retrieved response with citations from knowledge base
```

## Development

### Project Structure
```
ai-application/
    backend/
        app/
            chat/
                api.py              # Chat endpoint
                models.py           # Database models
                cruds.py            # Database operations
                chat_agent/
                    agent.py        # LangGraph orchestration
                    state.py        # Agent state management
                    nodes/          # Individual agent components
            indexing/
                api.py              # Indexing endpoint
                indexer.py          # Vector database operations
                schemas.py          # Request/response models
            core/
                db.py               # Database configuration
            main.py                 # FastAPI application
        .env                        # Environment variables
    frontend/
        main.py                     # Streamlit application
        chat.py                     # Chat interface component
    README.md
```

### Key Components

**Chat Agent Nodes:**
- `intent_router.py`: RAG vs. direct response routing
- `retriever.py`: Document search and filtering
- `generator.py`: Response generation with citations
- `generation_evaluator.py`: Quality control and validation
- `simple_assistant.py`: Direct responses without retrieval
- `fallback.py`: Error handling and graceful degradation

### Database Schema

**Users Table:**
- `id`: Primary key
- `username`: Unique identifier
- `messages`: Relationship to messages

**Messages Table:**
- `id`: Primary key
- `user_id`: Foreign key to users
- `message`: Content text
- `type`: 'user' or 'assistant'
- `timestamp`: Creation time

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI operations
- `SQLALCHEMY_DATABASE_URL`: Database connection (default: SQLite)

### Customization Options
- **Model Selection**: Update model names in node files
- **Collection Names**: Modify ChromaDB collection identifiers
- **Retry Limits**: Adjust iteration limits in router functions
- **Response Length**: Configure max tokens in generation prompts

## Troubleshooting

### Common Issues

**"No module named 'app'" Error:**
- Ensure you're running commands from the correct directory
- Verify the virtual environment is activated

**"OPENAI_API_KEY not found" Error:**
- Check that `.env` file exists in the `backend/` directory
- Verify the API key is valid and properly formatted

**"No such table: messages" Error:**
- The database tables are created automatically on startup
- Restart the backend server to trigger table creation

**ChromaDB Collection Issues:**
- Delete the `chroma` folder to reset the vector database
- Re-run the indexing process to repopulate

### Performance Considerations
- First-time indexing may take several minutes due to AI summarization
- Vector similarity search scales with document collection size
- Consider implementing pagination for large document sets

## Monitoring and Logging

The application includes comprehensive logging:
- **API Requests**: FastAPI automatic logging
- **Agent Decisions**: Router and evaluation outputs
- **Database Operations**: SQLAlchemy query logging
- **Error Tracking**: Exception handling with context

## Deployment Considerations

For production deployment:
1. **Environment**: Use production-grade ASGI server (Gunicorn + Uvicorn)
2. **Database**: Migrate from SQLite to PostgreSQL
3. **Vector Store**: Consider hosted ChromaDB or Pinecone
4. **API Keys**: Use secure secret management
5. **Monitoring**: Add APM tools and health checks

## Contributing

This project demonstrates advanced RAG architecture patterns:
- Multi-step agent reasoning with state management
- Quality control through iterative evaluation
- Graceful fallback handling for edge cases
- Separation of concerns with modular design
- Comprehensive error handling and user feedback