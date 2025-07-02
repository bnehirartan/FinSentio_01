# FinSentio: AI-Powered Financial Assistant

FinSentio is an intelligent financial assistant designed to support users with real-time market insights, risk analysis, document processing, and financial education. It leverages state-of-the-art Large Language Model (LLM) technologies to deliver personalized, data-driven responses in both Turkish and English. 

![image](https://github.com/user-attachments/assets/a3828da3-f6c0-4168-9676-6a3a8382ab69)
![image](https://github.com/user-attachments/assets/c7f333a8-d0f3-4c4e-a53d-fda82b2dbf51)
![image](https://github.com/user-attachments/assets/70272d33-5377-4046-b24d-7bcca993479e)


## Project Overview
![image](https://github.com/user-attachments/assets/92dce963-530d-44a5-b93d-941276cadaea)
![image](https://github.com/user-attachments/assets/fd9033fd-2475-47f9-81ce-d774abfbcf19)

This project implements an advanced LLM-based Financial Advisor and Education Assistant system using cutting-edge AI techniques, including:

  - Function Calling (non-parametric grounding)
  - Multi-Agent Architecture
  - RAG (Retrieval-Augmented Generation)
  - Prompt Engineering

These techniques empower the system to:
- Access and use up-to-date, real-world data
- Divide responsibilities between specialized agents
- Dynamically construct prompts tailored to user intent
- Provide source-backed, explainable financial answers

### Key Features
- Risk Assessment: Summarized in both table and natural language formats
- Real-Time Market Insights: Powered by APIs like Finnhub and FRED
- PDF-Based Financial Document Analysis
- Bilingual Support: Turkish 🇹🇷 and English 🇬🇧
- Smart Prompting: Intent classification and tool selection for each query
- LLM Cost Tracking: Real-time token usage and pricing calculator

## Core Capabilities

- Financial data retrieval and analysis
- Investment and risk advice based on the saved user profile
- Veri toplama ve analiz
- Secure and modular PDF document processing
- PDF dosya işleme
- ChromaDB-based persistent knowledge storage
- Transparent usage cost calculation for LLM queries

## Installation


```bash
pip install -r requirements.txt
```


1. Clone the repository:
```bash
git clone [proje-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python database.py
```

## Usage

Run the main app locally:

```bash
python app.py
```

The app will launch a web interface (via Gradio) where users can interact with the system, fill out a profile, and receive real-time insights.

## Project Structure

- `app.py`:  Main application entry point. Launches the UI and integrates all system modules.
- `financial_assistant.py`: Financial education module and agent coordination
- `riskanalyzer.py`: Risk analysis tools, report generation, and the main part that generates the response to the user
- `dataretrieval.py`: API integrations, data processing, and tool building for the function calling
- `database.py`: User & session management via SQLite
- `llm_cost_calculator.py`: Gemini token cost estimation
- `ChromaDBData/`: ChromaDB persisted documents
- `chroma_db/`: ChromaDB configuration and embeddings
- `requirements.txt`: Python dependencies
-  `users.db`: SQLite database file (auto-generated)

## Database

This project uses a lightweight SQLite database (users.db) for storing user information, login data, and risk profiles. No cloud database is required for local use.
