#!/usr/bin/env python3

__author__ = "xi"

import httpx
import numpy as np
from agent_types.common import LLMConfig
from agent_types.common import NDArray
from agent_types.retrieval import DenseEmbeddingRequest
from agent_types.retrieval import DenseEmbeddingResponse
from libentry.mcp.client import APIClient
from openai import OpenAI


class EmbeddingAdapter:
    """Adapter for various embedding services, supporting both local API clients and OpenAI."""

    def __init__(self, url: str | LLMConfig):
        """Initialize the EmbeddingAdapter.

        Args:
            url: The URL string for a local embedding service or an LLMConfig for OpenAI.
        """
        self.url = url

    def embedding(self, request: DenseEmbeddingRequest) -> DenseEmbeddingResponse:
        """Generate embeddings for the given text request.

        Args:
            request: The request containing text to embed and normalization settings.

        Returns:
            A DenseEmbeddingResponse containing the generated embeddings as an NDArray.
        """
        if isinstance(self.url, str):
            with APIClient(self.url) as client:
                return DenseEmbeddingResponse.model_validate(client.post(request))
        else:
            with OpenAI(
                    base_url=self.url.base_url,
                    api_key=self.url.api_key,
                    http_client=httpx.Client(verify=False)
            ) as client:
                response = client.embeddings.create(
                    model=self.url.model,
                    input=request.text,
                )

            data = response.data
            data.sort(key=lambda _i: _i.index)
            data = [
                [float(val) for val in row.embedding]
                for row in data
            ]
            if isinstance(request.text, str):
                data = data[0]

            emb = np.array(data, dtype=np.float32)
            if request.normalize:
                emb /= (np.linalg.norm(emb, 2, -1, keepdims=True) + 1e-30)
            emb = NDArray.from_array(emb)
            return DenseEmbeddingResponse(trace_id=request.trace_id, embedding=emb)
