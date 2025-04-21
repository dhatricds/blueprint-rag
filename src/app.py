import streamlit as st
from PIL import Image
import os
from pathlib import Path
import json
import sys
import torch
import logging
import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging with a more specific format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure PyTorch
torch.set_grad_enabled(False)  # Disable gradients
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.rag import RAGPipeline
from src.rag.types import QueryType

# Initialize storage paths
STORAGE_DIR = Path("storage")
VECTOR_DB_PATH = STORAGE_DIR / "vector_db.index"
CONTEXT_DB_PATH = STORAGE_DIR / "context_store.json"

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the RAG pipeline
@st.cache_resource(show_spinner=False)
def get_pipeline():
    try:
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(
            vector_db_path=str(VECTOR_DB_PATH),
            context_db_path=str(CONTEXT_DB_PATH)
        )
        logger.info("RAG pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return None

def process_query_results(results):
    """Process and display query results."""
    if not results:
        st.warning("No results found.")
        return
        
    st.subheader("Results")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"Result {i}", expanded=i==1):
            # Display the main text - handle both direct text and metadata text
            st.markdown(f"**Text:**")
            text = result.get("text", "")
            if not text and "metadata" in result:
                text = result["metadata"].get("text", "No text available")
            st.info(text)
            
            # Display metadata if available
            if metadata := result.get("metadata", {}):
                st.markdown("**Metadata:**")
                
                # Display count information for count queries
                if "count" in metadata:
                    st.metric("Count", metadata["count"])
                    if target := metadata.get("target_entity"):
                        st.write(f"Target: {target}")
                
                # Display location information
                if location := metadata.get("location"):
                    st.write(f"Location: {location}")
                
                # Display other metadata
                other_metadata = {k: v for k, v in metadata.items() 
                                if k not in ["count", "target_entity", "location", "text"]}
                if other_metadata:
                    st.json(other_metadata)
            
            # Display relevance score
            st.caption(f"Relevance Score: {1 - result.get('distance', 0):.2%}")

def main():
    st.set_page_config(
        page_title="Blueprint RAG System",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("Blueprint RAG System 🏗️")
    
    # Initialize pipeline
    with st.spinner("Initializing RAG pipeline..."):
        pipeline = get_pipeline()
        
    if pipeline is None:
        st.error("Failed to initialize the RAG pipeline. Please check the logs for details.")
        return
        
    st.success("RAG Pipeline initialized successfully!")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Blueprints")
        uploaded_file = st.file_uploader(
            "Choose a blueprint image",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Upload a blueprint image to process"
        )
        
        if uploaded_file:
            try:
                # Process the uploaded file
                image = Image.open(uploaded_file)
                
                # Add metadata
                metadata = {
                    "filename": uploaded_file.name,
                    "type": "blueprint",
                    "size": f"{image.size[0]}x{image.size[1]}",
                    "upload_time": str(pd.Timestamp.now())
                }
                
                # Process blueprint
                with st.spinner("Processing blueprint..."):
                    pipeline.process_blueprint(image, metadata)
                st.success("Blueprint processed successfully!")
                
            except Exception as e:
                logger.error(f"Error processing blueprint: {str(e)}", exc_info=True)
                st.error("Error processing blueprint. Please check the file and try again.")
    
    # Main area for querying
    st.header("Query Blueprints")
    
    # Query input
    query = st.text_input(
        "Enter your query",
        placeholder="e.g., How many bathrooms are there?"
    )
    
    # Example queries with descriptions
    st.markdown("### Example Queries")
    
    example_queries = {
        "Count Queries": [
            "How many bathrooms are there?",
            "Count the recessed light fixtures",
            "Number of windows in the living room"
        ],
        "Dimension Queries": [
            "What are the dimensions of the walk-in closet?",
            "What is the size of the master bedroom?",
            "How big is the garage?"
        ],
        "Location Queries": [
            "What is on the first floor?",
            "What features are in the basement?",
            "Show me what's in the kitchen"
        ],
        "General Queries": [
            "Tell me about the garage",
            "Describe the kitchen layout",
            "What type of flooring is used?"
        ]
    }
    
    # Display example queries in tabs
    tabs = st.tabs(list(example_queries.keys()))
    for tab, (category, examples) in zip(tabs, example_queries.items()):
        with tab:
            cols = st.columns(len(examples))
            for i, (col, example) in enumerate(zip(cols, examples)):
                if col.button(f"Try", key=f"example_{category}_{i}"):
                    query = example
                    st.session_state.query = example
    
    # Process query
    if query:
        try:
            with st.spinner("Processing query..."):
                results = pipeline.query(query)
                process_query_results(results)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error("Error processing query. Please try again.")

if __name__ == "__main__":
    main()