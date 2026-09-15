"""Public data models for ModelPack artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .constants import WEIGHT_RAW_MEDIA_TYPE


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    knowledge_cutoff: str | None = None
    reasoning: bool | None = None
    tool_usage: bool | None = None
    reward: bool | None = None
    languages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "inputTypes": list(self.input_types),
                "outputTypes": list(self.output_types),
                "knowledgeCutoff": self.knowledge_cutoff,
                "reasoning": self.reasoning,
                "toolUsage": self.tool_usage,
                "reward": self.reward,
                "languages": list(self.languages),
            }
        )


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    created_at: str | None = None
    authors: tuple[str, ...] = ()
    vendor: str | None = None
    family: str | None = None
    name: str | None = None
    version: str | None = None
    title: str | None = None
    description: str | None = None
    doc_url: str | None = None
    source_url: str | None = None
    datasets_url: tuple[str, ...] = ()
    revision: str | None = None
    licenses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "createdAt": self.created_at,
                "authors": list(self.authors),
                "vendor": self.vendor,
                "family": self.family,
                "name": self.name,
                "version": self.version,
                "title": self.title,
                "description": self.description,
                "docURL": self.doc_url,
                "sourceURL": self.source_url,
                "datasetsURL": list(self.datasets_url),
                "revision": self.revision,
                "licenses": list(self.licenses),
            }
        )


@dataclass(frozen=True, slots=True)
class ModelTechnicalConfig:
    architecture: str | None = None
    format: str | None = None
    param_size: str | None = None
    precision: str | None = None
    quantization: str | None = None
    capabilities: ModelCapabilities | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "architecture": self.architecture,
                "format": self.format,
                "paramSize": self.param_size,
                "precision": self.precision,
                "quantization": self.quantization,
                "capabilities": (
                    self.capabilities.to_dict() if self.capabilities else None
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ModelPackConfig:
    descriptor: ModelDescriptor = field(default_factory=ModelDescriptor)
    config: ModelTechnicalConfig = field(default_factory=ModelTechnicalConfig)

    def to_dict(self, diff_ids: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "config": self.config.to_dict(),
            "modelfs": {"type": "layers", "diffIds": list(diff_ids)},
        }


@dataclass(frozen=True, slots=True)
class ModelLayer:
    path: str | Path
    media_type: str = WEIGHT_RAW_MEDIA_TYPE
    artifact_path: str | None = None
    annotations: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PushResult:
    reference: str
    digest: str | None
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PullResult:
    reference: str
    files: tuple[Path, ...]
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
