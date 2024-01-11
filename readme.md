# **Dentsu RAG**

## **Description**

This LangChain RAG (Retrieval Augmented Generation) application is a state-of-the-art AI solution for processing and analyzing text data. It combines the efficiency of Redis as a vector database with the power of OpenAI's GPT models for generating human-like responses. The application is designed to process public financial PDF documents, like Nike’s 10k filings, and provide rich, context-aware responses.

## **Features**

- **PDF to Text Conversion**: Utilizes `UnstructuredFileLoader` and `ocrmac` for accurate OCR processing of PDF documents.
- **Redis Vector Database**: Employs Redis for real-time context retrieval and efficient data management.
- **Generative AI Responses**: Integrates OpenAI’s `gpt-3.5-turbo-16k` LLM for advanced text generation.
- **AI-Powered Analysis**: Combines context from source documents with AI capabilities for comprehensive analysis.

## **Getting Started**

### **Prerequisites**

- Python 3.9
- Redis Cloud or local Redis Stack
- OpenAI API key

### **Local Redis Stack Setup**

1. **Install Redis Stack**:\
   https://redis.io/docs/install/install-stack/mac-os/

   `brew tap redis-stack/redis-stack brew install redis-stack`

2. **Start Redis Stack Server**:

   `redis-stack-server`

### **Application Setup**

1. **Environment Setup**: Set your environment variables (replace `<...>` with actual values):

   `export OPENAI_API_KEY=<your_openai_api_key> export REDIS_HOST=<your_redis_host> export REDIS_PORT=<your_redis_port> export REDIS_USER=<your_redis_user> # Optional export REDIS_PASSWORD=<your_redis_password>`

   Or set `REDIS_URL` directly.

2. **Python Environment**: Create and activate a Python virtual environment:

   `python3.9 -m venv lc-template source lc-template/bin/activate`

3. **Install LangChain CLI**:

   `pip install -U langchain-cli pydantic==1.10.13`

4. **Create LangChain Project**:

   `langchain app new test-rag`

   Follow the prompt to install the template.

5. **Enter Project Directory**:

   `cd test-rag`

6. **Ingest Data**: Place your PDFs in `data/` and run:

   `cd packages/rag-redis python ingest.py`

7. **Serve the Application**: Go back to the root and start the server:

   `cd ../ && cd ../ langchain serve`

8. **Access the API**:

   - Documentation: <http://127.0.0.1:8000/docs>
   - Playground: <http://127.0.0.1:8000/rag-redis/playground>
