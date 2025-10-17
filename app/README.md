# IT Service Classifier and Ticket Assistant

This project is a **FastAPI application** for **triaging IT service desk issues**.
It combines **machine learning classifiers**, **retrieval-augmented knowledge lookups**, and an **LLM-powered assistant** to help route and troubleshoot tickets.

The system brings together three core components:

1. **Ticket Classifier** – a fine-tuned Transformer model (or API) that predicts ServiceNow assignment groups from ticket text.
2. **Knowledge Retrieval** – integrations with Confluence (assignment groups page) and Qdrant (historical tickets) to provide team details, escalation paths, and related cases.
3. **Conversational Assistant** – an LLM-driven chat interface that explains predictions, surfaces relevant knowledge, and supports interactive triage, with session history stored in Redis or in memory.

---

## Project Structure

```
app/
├── api/                 # API routes (FastAPI routers)
│   ├── routes_home.py   # Frontend form and home page
│   ├── routes_classify.py
│   ├── routes_chat.py
│   ├── routes_health.py
│   └── __init__.py
├── assets/
│   ├── models/          # Classifier model, tokenizer, and label mapping
│   ├── static/          # Static assets (e.g., JavaScript, CSS)
│   └── templates/       # HTML templates (Jinja2)
├── core/                # Configuration and shared utilities
│   ├── config.py
│   ├── dependencies.py
│   ├── core_utils.py
│   └── __init__.py
├── prompts/             # LLM prompt templates - maybe move this to assets
│   ├── prompt1.txt
│   ├── prompt2.txt
│   └── etc.
├── schemas/             # Pydantic request/response models
│   ├── classify.py
│   ├── chat.py
│   ├── query.py
│   ├── ticket.py
│   ├── info_form.py
│   ├── debug.py
│   └── __init__.py
├── services/            # Business logic and integrations
│   ├── classifier_service.py
│   ├── llm_client.py
│   ├── session_service.py
│   ├── service_utils.py
│   ├── rag/             # Retrieval-augmented generation helpers
│   |   ├── groups_query.py
│   |   ├── ticket_query.py
│   |   └── __init__.py
│   └── workflows/   # Workflow orchestration
│       ├── escalation_workflow.py
│       ├── troubleshooting_workflow.py
│       ├── router.py
│       └── __init__.py
├── app.py               # FastAPI application entrypoint
├── Dockerfile
└── requirements.txt
```

Here’s an example of what the `assets/models/` directory should look like:
```
models/
  itsclassifier/
    conversions/         # Label mappings (e.g., id2label.json)
    model/               # Core model weights and config (config.json, model.safetensors)
    tokenizer/           # Tokenizer vocab, special tokens, and tokenizer config
```
---

## Configuration

Application behavior is controlled via environment variables.

---

## Running Locally

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis and Qdrant**

   ```bash
   docker run -d --name redis -p 6379:6379 redis:7

   docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest
   ```

3. **Start the application**
  Note that you will need to setup enviorment variables for correct functionality.

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8080
   ```

4. Access the application:

   * Web interface: `http://localhost:8080/`

---

## API Endpoints

| Path            | Method | Description                                      |
| --------------- | ------ | ------------------------------------------------ |
| `/`             | GET    | Render ticket submission form (HTML)             |
| `/`             | POST   | Submit ticket and run workflow                   |
| `/chat`         | POST   | LLM chat endpoint for UI (JSON request/response) |
| `/api/classify` | POST   | Classify ticket (JSON request/response)          |
| `/api/health`   | GET    | Health check for readiness/liveness probes       |

---

## Docker

Build and run the container:

```bash
docker build -t ticket-assistant .
docker run -p 8080:8080 --env-file .env ticket-assistant
```

Or with all DBs included:
```bash
docker compose up -d
```
---

