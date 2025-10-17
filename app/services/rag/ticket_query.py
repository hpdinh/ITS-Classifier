from typing import Optional, Literal
from qdrant_client.http import models
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding

from services import LLMClient
from core.config import QdrantSettings, PromptSettings

import logging
logger = logging.getLogger(__name__)

class SnowTicketQuery:
    """
    Wrapper around Qdrant for retrieving ServiceNow tickets using dense or sparse embeddings.

    Provides two retrieval modes:
      - Dense search: Uses LLM-generated embeddings for semantic similarity.
      - Sparse search: Uses BM25-style sparse embeddings for keyword-based retrieval.

    Attributes:
        client (QdrantClient): Qdrant client for interacting with the vector database.
        collection_name (str): Name of the Qdrant collection used for storing tickets.
        query_task (str): Prompt text used for query augmentation before embedding.
        sparse_model (SparseTextEmbedding): BM25 sparse embedding generator.
        dense_k (int): Default number of top results for dense search.
        sparse_k (int): Default number of top results for sparse search.
        llm_client (Optional[LLMClient]): LLM client used for rephrasing and dense embeddings.
    """

    def __init__(
        self,
        settings: QdrantSettings = QdrantSettings(),
        prompt_settings: PromptSettings = PromptSettings(),
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize a SnowTicketQuery instance.

        Args:
            settings (QdrantSettings, optional): Configuration for connecting to Qdrant
                (URL, API key, collection name, top_k defaults).
            prompt_settings (PromptSettings, optional): Settings for loading query prompts.
            llm_client (Optional[LLMClient]): LLM client used to rephrase queries
                and generate dense embeddings. Required for dense search.
        """
        self.client = QdrantClient(url=settings.url, api_key = settings.api_key)
        self.collection_name = settings.collection_name
        self.query_task = prompt_settings.load_prompt(prompt_settings.query_prompt_file)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.dense_k = settings.dense_top_k
        self.sparse_k = settings.sparse_top_k
        self.llm_client = llm_client

    async def dense_search(self, query_text: str, top_k: Optional[int] = None, input_type: Literal['call', 'ticket'] = 'ticket'):
        """
        Perform a dense embedding search over ServiceNow tickets.

        The query is first rephrased using the LLM client, then embedded,
        and finally matched against the Qdrant collection.

        Args:
            query_text (str): The raw user query (ticket description or call text).
            top_k (Optional[int], optional): Number of top results to return.
                Defaults to self.dense_k if not provided.
            input_type (Literal['call', 'ticket'], optional): Type of input text,
                which determines rephrasing behavior. Defaults to 'ticket'.

        Returns:
            List[models.ScoredPoint]: A list of matching Qdrant points with scores.

        Raises:
            ValueError: If the generated dense embedding does not match the expected dimension (2560).
        """
        rephrased = await self.llm_client.rephrase(query_text, mode=input_type)
        query_emb = (await self.llm_client.embed(self.query_task + rephrased))[0]
        if len(query_emb) != 2560:
            raise ValueError(f"Embedding size {len(query_emb)} != expected 2560")
        logger.info(f"Embedding length: {len(query_emb)}")
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            using="dense",
            limit=top_k or self.dense_k,
        ).points
        return results

    async def sparse_search(self, query_text: str, top_k: Optional[int] = None, input_type = Literal['call', 'ticket']):
        """
        Perform a sparse BM25 embedding search over ServiceNow tickets.

        The query is rephrased using the LLM client, then converted into a
        sparse vector and matched against the Qdrant collection.

        Args:
            query_text (str): The raw user query (ticket description or call text).
            top_k (Optional[int], optional): Number of top results to return.
                Defaults to self.sparse_k if not provided.
            input_type (Literal['call', 'ticket'], optional): Type of input text,
                which determines rephrasing behavior. Defaults to 'ticket'.

        Returns:
            List[models.ScoredPoint]: A list of matching Qdrant points with scores.
        """
        rephrased = await self.llm_client.rephrase(query_text, mode=input_type)
        sparse_vec = list(self.sparse_model.embed([rephrased]))[0]

        sparse_qdrant = models.SparseVector(
            indices=sparse_vec.indices,
            values=sparse_vec.values,
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_qdrant,
            using="sparse",
            limit=top_k or self.sparse_k,
        ).points
        return results
