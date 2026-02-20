# ISL Chatbot Backend

This is the FastAPI backend for the Indian Sign Language (ISL) chatbot. It is powered by LangChain and Google's Gemini (gemini-2.5-flash) to process user prompts, retain conversational memory, and return highly visual, accessible responses in plain text.

## 🚀 Prerequisites

- Python 3.8+ installed on your system.
- A Google Gemini API Key.

## 🛠️ Setup & Installation

**1. Create a Virtual Environment**
It is highly recommended to run this project inside an isolated Python virtual environment.

```bash
# Create the environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Setup environment variables**

```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

**4. Run the server**

```bash
uvicorn main:app --reload
```
