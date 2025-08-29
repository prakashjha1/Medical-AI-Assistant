# 🏥 Medical Assistant

An intelligent medical assistant powered by RAG (Retrieval Augmented Generation) that provides accurate medical information using a curated knowledge base and Google's Gemini-2.5-Flash.


<img width="1909" height="915" alt="Screenshot 2025-08-29 220942" src="https://github.com/user-attachments/assets/4c9f3f15-1e1e-421c-b2b7-686db824f778" />

## ✨ Features

- **🤖 Intelligent Responses**: Powered by Google's Gemini-2.5-Flash for accurate medical information
- **📚 Knowledge Base**: Uses Pinecone vector database for reliable medical document retrieval
- **💬 Chat Interface**: User-friendly Streamlit chat interface
- **🧠 Memory**: Remembers conversation context using LangChain buffer memory
- **🔒 Secure**: Environment-based API key management
- **🐳 Containerized**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Google API key
- Pinecone API key

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd medical-assistant
```

### 2. Set Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_openai_api_key_here
```

### 3. Run with Docker Compose

```bash
# Start the application
docker-compose --env-file .env up -d

# View logs
docker-compose logs -f
```
### 4. Access the Application

Open your browser and go to: **http://localhost:8501**

**Large Docker image size:**
- Current size: ~20GB (includes embedding model and dependencies)
- Consider using Docker multi-stage builds for production
