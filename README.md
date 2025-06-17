# Blood Donation Chatbot

A specialized chatbot system for the Blood Donation platform that provides information about blood donation, answers common questions, and assists users with blood donation-related inquiries using Google's Gemini AI model with RAG (Retrieval-Augmented Generation).

## Features

- **FastAPI Backend**: High-performance asynchronous API with automatic documentation
- **Gemini AI Integration**: Leverages Google's Gemini model for intelligent, contextually-relevant responses
- **RAG Capabilities**: Enhanced responses through Retrieval-Augmented Generation using FAISS vector store
- **PDF Document Processing**: Processes blood donation documents for accurate domain knowledge
- **Context-Aware Responses**: Maintains conversation context for natural dialog flow
- **CORS Support**: Seamless integration with the Blood Donation frontend application

## Project Structure

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoint.py              # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                # Configuration settings
│   ├── create_vector/
│   │   └── create_vectorstore.py    # Vector store creation and management
│   ├── data/                        # Blood donation documents directory
│   ├── models/                      # Pydantic data models
│   ├── services/                    # Core chatbot services
│   ├── vectorstore/                 # FAISS vector store for document embeddings
│   └── utils/                       # Utility functions and helpers
├── env/                             # Python virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key (https://ai.google.dev/)

### 1. Environment Setup

1. Clone the repository or navigate to the chatbot directory
2. Create a virtual environment (if not already created):

```bash
python -m venv env
```

3. Activate the virtual environment:

**Windows:**

```bash
env\Scripts\activate
```

**Linux/Mac:**

```bash
source env/bin/activate
```

4. Create a `.env` file in the root directory with the following:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Knowledge Base

Place your blood donation PDF documents in the `app/data/` directory. These will be processed to create the chatbot's knowledge base.

### 4. Create Vector Store

```bash
python -m app.create_vector.create_vectorstore
```

This command will:

- Process the PDF documents in the data directory
- Extract text content
- Generate embeddings using the specified embedding model
- Store the vector embeddings in the FAISS vector store

### 5. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API Endpoint**: http://localhost:8000
- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### Chat Endpoint

**POST** `/api/chat`

Send a message to the chatbot and receive an AI-generated response.

#### Request Body:

```json
{
  "message": "What are the requirements for blood donation?",
  "include_sources": true
}
```

#### Response:

```json
{
  "message": "To donate blood, you typically need to meet these requirements: be at least 17 years old (16 with parental consent in some states), weigh at least 110 pounds, be in good general health, and pass a brief health screening. You should wait at least 8 weeks between whole blood donations and have adequate iron levels...",
  "sources": [
    {
      "content": "Blood donation requirements include minimum age of 17 (16 with parental consent in some areas), weight of at least 110 pounds, good health status...",
      "metadata": {
        "source": "donation_requirements.pdf",
        "page": 2
      }
    }
  ],
  "timestamp": "2025-06-17T10:30:00Z"
}
```

## Configuration

Key configuration options in `app/core/config.py`:

- `GEMINI_MODEL`: Gemini model to use (default: "gemini-pro")
- `GEMINI_TEMPERATURE`: Controls response creativity (0.0-1.0)
- `GEMINI_MAX_OUTPUT_TOKENS`: Maximum response length
- `EMBEDDING_MODEL`: Model for document embeddings
- `RETRIEVAL_K`: Number of documents to retrieve for context

## Usage Examples

### Python Client Example:

```python
import requests

# Send a chat message
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "message": "What is the process of blood donation?",
        "include_sources": True
    }
)

data = response.json()
print(f"Bot: {data['message']}")

# If sources were included
if "sources" in data:
    print("\nSources:")
    for source in data["sources"]:
        print(f"- {source['metadata'].get('source', 'Unknown source')}")
```

### cURL Example:

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the benefits of regular blood donation?",
    "include_sources": true
  }'
```

## Development and Extending the Chatbot

### Adding New Documents

1. Add new PDF files to the `app/data/` directory
2. Run the vector store creation script:
   ```bash
   python -m app.create_vector.create_vectorstore
   ```

### Customizing the Chatbot

The chatbot's behavior can be customized by:

1. Modifying prompt templates in the chatbot service
2. Adjusting retrieval parameters for the RAG system
3. Fine-tuning temperature and other model parameters in `config.py`
4. Adding specialized handlers for specific types of blood donation queries

### Troubleshooting

If you encounter issues:

1. Check the application logs for detailed error information
2. Verify your Gemini API key is correct and has sufficient quota
3. Ensure the vector store has been properly created with relevant documents
4. Check that all dependencies are correctly installed

## Integration with Blood Donation Platform

This chatbot is designed to integrate with the Blood Donation platform, providing users with accurate information about:

- Blood donation eligibility
- Donation process and what to expect
- Health benefits of donating blood
- Blood types and compatibility
- Upcoming blood donation events
- Post-donation care and recommendations

### Running in Development Mode:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests:

```bash
# Add your test files to a tests/ directory
pytest tests/
```

## Production Deployment

1. Set `ENVIRONMENT=production` in your `.env` file
2. Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Troubleshooting

### Common Issues:

1. **Gemini API Key Issues**: Ensure your API key is correctly set in the `.env` file
2. **Vector Store Not Found**: Run `python create_vectorstore.py` to create the knowledge base
3. **PDF Loading Errors**: Ensure PDF files are in the `data/` directory and are readable
4. **Memory Issues**: Reduce `RETRIEVAL_K` or chunk size if running on limited resources

## License

This project is intended for educational and development purposes. Please ensure you have appropriate licenses for any medical documents used in the knowledge base.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please check the logs and ensure all dependencies are properly installed.
