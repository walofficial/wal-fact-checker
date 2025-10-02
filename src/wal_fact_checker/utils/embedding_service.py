import logging
from typing import Literal

from google.genai import Client, types

from wal_fact_checker.core.settings import settings

logger = logging.getLogger(__name__)

gemini_client = Client(api_key=settings.gcp_genai_key)


class EmbeddingService:
    """Service for generating title embeddings using Gemini text-embedding-001 model"""

    EMBEDDING_DIMENSIONS = 3072

    async def generate_embeddings(
        self,
        texts: list[str],
        task_type: Literal[
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        ]
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts asynchronously"""
        if not texts:
            return []

        try:
            response = await gemini_client.aio.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.EMBEDDING_DIMENSIONS,
                    task_type=task_type,
                ),
            )

            logger.info(
                "Generated embeddings for texts",
                extra={
                    "json_fields": {
                        "texts_count": len(texts),
                        "operation": "generate_embeddings",
                    },
                    "labels": {"component": "embedding_service"},
                },
            )

            return [embedding.values for embedding in response.embeddings]

        except Exception as e:
            logger.exception(
                "Failed to generate embeddings",
                extra={
                    "json_fields": {
                        "error": str(e),
                        "texts_count": len(texts),
                        "operation": "generate_embeddings",
                    },
                    "labels": {"component": "embedding_service", "severity": "high"},
                },
            )
            # Return zero vectors as fallback
            return [[0.0] * self.EMBEDDING_DIMENSIONS for _ in texts]


embedding_service = EmbeddingService()
