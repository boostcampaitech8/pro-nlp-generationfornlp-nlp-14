"""Embedder 구현체들.

EmbedderProtocol을 구현하는 외부 서비스 클라이언트들.
"""

from infrastructure.embedders.solar import SolarEmbedder, SolarEmbedderConfig

__all__ = [
    "SolarEmbedder",
    "SolarEmbedderConfig",
]
