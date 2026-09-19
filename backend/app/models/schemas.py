"""
Pydantic schemas for request/response models.

This file will be populated in later phases as endpoints are implemented.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str


class ChunkMetadata(BaseModel):
    """Metadata associated with each text chunk stored in Pinecone.

    Pinecone metadata values are strictly restricted to strings, numbers (int/float),
    booleans, or lists of strings. All fields in this model conform directly to these types.
    """

    document_id: str
    chunk_id: str
    page_number: int
    source_text: str
    char_start: int
    char_end: int

    def to_pinecone_metadata(self) -> dict[str, str | int]:
        """Serialize model to a Pinecone-compatible metadata dictionary.

        Ensures all keys and values conform strictly to Pinecone's metadata value constraints.
        """
        return self.model_dump()

    @classmethod
    def from_pinecone_metadata(cls, metadata: dict) -> "ChunkMetadata":
        """Deserialize a Pinecone metadata dictionary back into a validated ChunkMetadata instance."""
        return cls(**metadata)

