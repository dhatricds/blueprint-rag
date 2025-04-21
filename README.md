# Blueprint RAG System 🏗️

A Retrieval-Augmented Generation (RAG) system for analyzing and querying architectural blueprints. This system uses advanced NLP techniques to understand and answer questions about blueprint details.

## Features

- Upload and process architectural blueprints
- Query blueprint information using natural language
- Support for multiple query types:
  - Count queries (e.g., "How many bathrooms are there?")
  - Dimension queries (e.g., "What are the dimensions of the walk-in closet?")
  - Location queries (e.g., "What is on the first floor?")
  - General queries (e.g., "Describe the kitchen layout")
- Vector-based similarity search using FAISS
- Context-aware responses with metadata support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dhatricds/blueprint-rag.git
cd blueprint-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to http://localhost:8501

3. Upload blueprint images using the sidebar

4. Enter queries in the main area or try the example queries

## Project Structure

```
.
├── src/
│   ├── app.py                 # Streamlit web application
│   └── rag/
│       ├── __init__.py
│       ├── embeddings/        # Embedding generation
│       ├── ocr/              # OCR processing
│       ├── pipeline.py       # Main RAG pipeline
│       ├── query_processor/  # Query processing
│       ├── storage/         # Vector and context storage
│       └── types.py         # Type definitions
├── tests/                   # Test files
├── storage/                # Storage for vector and context DBs
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.