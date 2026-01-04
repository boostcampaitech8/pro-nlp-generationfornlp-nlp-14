"""Infrastructure 계층.

외부 서비스(API 클라이언트)들의 구현체를 제공합니다.
모든 구현체는 core.protocols의 Protocol을 따릅니다.
"""

from infrastructure.embedders import SolarEmbedder, SolarEmbedderConfig

__all__ = [
    # Embedders
    "SolarEmbedder",
    "SolarEmbedderConfig",
]
