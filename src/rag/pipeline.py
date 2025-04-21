"""RAG pipeline for processing blueprints and queries."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import re
from collections import Counter

from .storage.blueprint_vector_db import BlueprintVectorDB
from .storage.context_store import ContextStore
from .embeddings import get_embeddings
from .ocr import extract_text
from .query_processor import QueryProcessor
from .types import QueryType, QueryIntent, QueryResult

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Pipeline for processing blueprints and queries."""
    
    def __init__(self, vector_db_path: str, context_db_path: str):
        """Initialize the RAG pipeline.
        
        Args:
            vector_db_path: Path to vector database
            context_db_path: Path to context database
        """
        try:
            logger.info("Initializing RAG pipeline")
            
            # Ensure parent directories exist
            vector_db_dir = Path(vector_db_path).parent
            context_db_dir = Path(context_db_path).parent
            vector_db_dir.mkdir(parents=True, exist_ok=True)
            context_db_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize vector database
            logger.info("Initializing vector database")
            self.vector_db = BlueprintVectorDB(
                db_path=vector_db_path,
                dimension=768  # BERT embedding dimension
            )
            
            # Initialize context store
            logger.info("Initializing context store")
            self.context_store = ContextStore(context_db_path)
            
            self.query_processor = QueryProcessor()
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a document to the RAG pipeline.
        
        Args:
            text: Document text
            metadata: Optional metadata to store with the document
        """
        try:
            # Get embeddings for the text
            embeddings = get_embeddings(text)
            
            # Add to vector database
            self.vector_db.add_vector(
                vector=embeddings,
                metadata={
                    'text': text,
                    'metadata': metadata or {}
                }
            )
            
            logger.info("Added document to RAG pipeline")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def _extract_dimensions(self, text: str) -> List[str]:
        """Extract dimension patterns from text.
        
        Args:
            text: Input text
            
        Returns:
            List of dimension strings found
        """
        # Match patterns like: 8x10, 8' x 10', 8ft x 10ft, etc.
        dimension_patterns = [
            r'\d+\s*[xX]\s*\d+',  # 8x10
            r'\d+\s*(?:ft|feet|\')\s*[xX]\s*\d+\s*(?:ft|feet|\')',  # 8ft x 10ft
            r'\d+\s*(?:ft|feet|\')\s*by\s*\d+\s*(?:ft|feet|\')',  # 8ft by 10ft
        ]
        
        dimensions = []
        for pattern in dimension_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            dimensions.extend(match.group(0) for match in matches)
        
        return dimensions

    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate text similarity score using multiple metrics.
        
        Args:
            query: Query text
            text: Document text
            
        Returns:
            Combined text similarity score
        """
        # Normalize texts
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Extract room type from query (e.g., "walk-in closet", "kitchen", etc.)
        room_type_match = re.search(r'([\w-]+(?:\s+[\w-]+)*(?:\s+closet|\s+room|\s+area)?)\s*(?:\d|$)', query_lower)
        room_type = room_type_match.group(1) if room_type_match else query_lower
        
        # Handle compound terms (e.g., "walk-in closet")
        compound_terms = {
            'walk-in closet': ['walk-in', 'walk in', 'walkin'],
            'living room': ['living', 'family room'],
            'dining room': ['dining'],
            'utility room': ['utility'],
            'recreation room': ['recreation', 'rec room']
        }
        
        # 1. Room type matching (50% weight)
        room_score = 0.0
        
        # Check for compound terms first
        for compound, variants in compound_terms.items():
            if compound in room_type or any(variant in room_type for variant in variants):
                # Check if the compound term or any of its variants are in the text
                if compound in text_lower or any(variant in text_lower for variant in variants):
                    room_score = 1.0
                    break
        
        # If no compound term match, try regular matching
        if room_score == 0.0:
            if room_type in text_lower:
                # Check if the room type appears as a complete phrase
                room_boundaries = r'\b' + re.escape(room_type) + r'\b'
                if re.search(room_boundaries, text_lower):
                    room_score = 1.0
                else:
                    # Partial match with word boundaries
                    room_parts = room_type.split()
                    matched_parts = sum(1 for part in room_parts 
                                     if re.search(r'\b' + re.escape(part) + r'\b', text_lower))
                    room_score = matched_parts / len(room_parts)
        
        # 2. Dimension matching (30% weight)
        dimension_score = 0.0
        query_dimensions = self._extract_dimensions(query_lower)
        text_dimensions = self._extract_dimensions(text_lower)
        
        if query_dimensions and text_dimensions:
            # If dimensions are specified in query, prioritize exact matches
            dimension_score = 1.0 if any(qd in text_dimensions for qd in query_dimensions) else 0.0
        elif text_dimensions:
            # If no dimensions in query but text has dimensions, give partial score
            dimension_score = 0.5
        
        # 3. Context relevance (20% weight)
        context_score = 0.0
        # Look for relevant context indicators
        context_indicators = {
            'closet': ['storage', 'shelves', 'hanging', 'clothes', 'wardrobe', 'walk-in', 'walk in'],
            'kitchen': ['counter', 'cabinets', 'sink', 'appliances', 'stove', 'island'],
            'bathroom': ['shower', 'toilet', 'sink', 'bath', 'vanity', 'en-suite'],
            'bedroom': ['bed', 'closet', 'window', 'master', 'guest'],
            'living': ['room', 'space', 'window', 'area', 'family']
        }
        
        # Find relevant indicators for the room type
        relevant_indicators = []
        for key, indicators in context_indicators.items():
            if key in room_type:
                relevant_indicators.extend(indicators)
        
        if relevant_indicators:
            matched_indicators = sum(1 for indicator in relevant_indicators if indicator in text_lower)
            context_score = min(1.0, matched_indicators / len(relevant_indicators))
        
        # Combine scores with weights
        final_score = (0.5 * room_score) + (0.3 * dimension_score) + (0.2 * context_score)
        
        logger.debug(f"Text similarity scores - Room: {room_score:.3f}, Dimension: {dimension_score:.3f}, Context: {context_score:.3f}, Final: {final_score:.3f}")
        
        return final_score

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query and return relevant results.
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            List of results with text, score, and metadata
        """
        try:
            # Get query intent
            query_intent = self.query_processor.process_query(query_text)
            query_type = query_intent.query_type
            
            logger.info(f"Query type detected: {query_type}")
            
            # Get initial results from vector database
            results = self.vector_db.search(
                query_text=query_text,
                k=k
            )
            
            # Process results based on query type
            if query_type == QueryType.COUNT:
                return self._process_count_query(query_text, results, query_intent)
            elif query_type == QueryType.LOCATION:
                return self._process_location_query(query_text, results, query_intent)
            elif query_type == QueryType.DIMENSION:
                return self._process_dimension_query(query_text, results, query_intent)
            else:
                return self._process_general_query(query_text, results)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _process_count_query(self, query_text: str, results: List[Dict[str, Any]], query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """Process a count query.
        
        Args:
            query_text: Original query text
            results: Initial search results
            query_intent: Parsed query intent
            
        Returns:
            Processed results with count information
        """
        target_entity = query_intent.target_entity
        location = query_intent.location
        
        # Filter results by location if specified
        if location:
            results = [r for r in results if location.lower() in r.get("metadata", {}).get("text", "").lower()]
        
        # Count occurrences of target entity
        count = sum(1 for r in results if target_entity.lower() in r.get("metadata", {}).get("text", "").lower())
        
        # Add count to results
        for r in results:
            r["count"] = count
            
        return results

    def _process_location_query(self, query_text: str, results: List[Dict[str, Any]], query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """Process a location query.
        
        Args:
            query_text: Original query text
            results: Initial search results
            query_intent: Parsed query intent
            
        Returns:
            Results filtered by location
        """
        location = query_intent.location
        
        # Filter results by location
        if location:
            results = [r for r in results if location.lower() in r.get("metadata", {}).get("text", "").lower()]
            
        return results

    def _process_dimension_query(self, query_text: str, results: List[Dict[str, Any]], query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """Process a dimension query.
        
        Args:
            query_text: Original query text
            results: Initial search results
            query_intent: Parsed query intent
            
        Returns:
            Results with dimension information
        """
        # Extract dimensions from results
        for r in results:
            text = r.get("metadata", {}).get("text", "")
            dimensions = self._extract_dimensions(text)
            r["dimensions"] = dimensions
            
        return results

    def _process_general_query(self, query_text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a general query.
        
        Args:
            query_text: Original query text
            results: Initial search results
            
        Returns:
            Processed results
        """
        # Calculate text similarity scores
        for r in results:
            text = r.get("metadata", {}).get("text", "")
            similarity_score = self._calculate_text_similarity(query_text, text)
            r["similarity_score"] = similarity_score
            
        # Sort by similarity score
        results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return results

    def process_blueprint(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process a blueprint image.
        
        Args:
            image: Blueprint image
            metadata: Optional metadata to store with the blueprint
            
        Returns:
            Extracted text from the blueprint
        """
        try:
            # Extract text from image
            text = extract_text(image)
            
            # Add to RAG pipeline
            self.add_document(text, metadata)
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing blueprint: {str(e)}")
            raise