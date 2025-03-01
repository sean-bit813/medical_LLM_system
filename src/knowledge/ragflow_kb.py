# src/knowledge/ragflow_kb.py
import requests
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class RAGFlowKnowledgeBase:
    """Knowledge Base implementation that uses the RAGFlow API for retrievals"""

    def __init__(self, api_url: str = None, api_key: str = None, dataset_ids: List[str] = None):
        """
        Initialize the RAGFlow knowledge base

        Args:
            api_url: Base URL for the RAGFlow API
            api_key: API key for authentication
            dataset_ids: List of dataset IDs to query
        """
        self.api_url = api_url or "http://ragflow-api-url/api/v1"
        self.api_key = api_key
        self.dataset_ids = dataset_ids or []

        # Validate initialization parameters
        if not self.api_key:
            logger.warning("No API key provided for RAGFlow. Authentication will fail.")
        if not self.dataset_ids:
            logger.warning("No dataset_ids provided. Queries will not return results.")

    def configure(self, api_url: str = None, api_key: str = None,
                  dataset_ids: List[str] = None, document_ids: List[str] = None):
        """
        Configure the RAGFlow client

        Args:
            api_url: Base URL for the RAGFlow API
            api_key: API key for authentication
            dataset_ids: List of dataset IDs to query
            document_ids: List of document IDs to query
        """
        if api_url:
            self.api_url = api_url
        if api_key:
            self.api_key = api_key
        if dataset_ids:
            self.dataset_ids = dataset_ids


    def search(self, query: str, k: int = 5, similarity_threshold: float = 0.2, rerank_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using the RAGFlow API

        Args:
            query: Query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            rerank_id: Rerank ID

        Returns:
            List of retrieved chunks with text and metadata
        """
        # Validate parameters
        if not query or query.strip() == "":
            logger.warning("Empty query provided to RAGFlow search")
            return []

        # Prepare the API request
        url = f"{self.api_url}/retrieval"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "question": query,
            "dataset_ids": self.dataset_ids,
            "similarity_threshold": similarity_threshold,
            "top_k": k,
            "rerank_id": rerank_id
        }

        try:
            # Make the API request
            logger.info(f"Sending retrieval request to RAGFlow: {query}")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Process the response
            result = response.json()

            if result.get("code") != 0:
                logger.error(f"RAGFlow API error: {result}")
                return []

            # Extract and format the chunks
            chunks = result.get("data", {}).get("chunks", [])
            formatted_results = []

            for chunk in chunks:
                formatted_results.append({
                    'text': chunk.get('content', ''),
                })

            logger.info(f"Retrieved {len(formatted_results)} chunks from RAGFlow")
            return formatted_results

        except requests.exceptions.RequestException as e:
            logger.error(f"RAGFlow API request failed: {e}")
            return []
        except ValueError as e:
            logger.error(f"Failed to parse RAGFlow API response: {e}")
            return []

    def add_dataset(self, dataset_id: str) -> None:
        """
        Add a dataset ID to the list of datasets to query

        Args:
            dataset_id: Dataset ID to add
        """
        if dataset_id and dataset_id not in self.dataset_ids:
            self.dataset_ids.append(dataset_id)
