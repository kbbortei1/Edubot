# EduBot API

AI-powered educational tutor API with vision support.

## Features
- Text-based Q&A
- Image understanding (diagrams, math problems, etc.)
- RAG (Retrieval Augmented Generation) for custom documents
- Web search integration

## API Endpoints

### POST /ask
```json
{
  "question": "What is photosynthesis?",
  "image": null,
  "use_rag": false,
  "use_web": false
}
```

## Environment Variables
- `GROQ_API_KEY` - Get from https://console.groq.com
- `SERPER_API_KEY` - Get from https://serper.dev

## Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
