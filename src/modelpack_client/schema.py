"""JSON schema of the CNCF ModelPack model config.

Copied from https://github.com/modelpack/model-spec/blob/main/schema/config-schema.json
"""

from typing import Any

CONFIG_SCHEMA: dict[str, Any] = {
    "description": "Model Artifact Configuration Schema",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/modelpack/model-spec/config",
    "type": "object",
    "properties": {
        "descriptor": {"$ref": "#/$defs/ModelDescriptor"},
        "modelfs": {"$ref": "#/$defs/ModelFS"},
        "config": {"$ref": "#/$defs/ModelConfig"},
    },
    "additionalProperties": False,
    "required": ["descriptor", "config", "modelfs"],
    "$defs": {
        "ModelConfig": {
            "type": "object",
            "properties": {
                "architecture": {"type": "string"},
                "format": {"type": "string"},
                "paramSize": {"type": "string"},
                "precision": {"type": "string"},
                "quantization": {"type": "string"},
                "capabilities": {"$ref": "#/$defs/ModelCapabilities"},
            },
            "additionalProperties": False,
        },
        "ModelDescriptor": {
            "type": "object",
            "properties": {
                "createdAt": {"type": "string", "format": "date-time"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "family": {"type": "string"},
                "name": {"type": "string", "minLength": 1},
                "docURL": {"type": "string"},
                "sourceURL": {"type": "string"},
                "datasetsURL": {"type": "array", "items": {"type": "string"}},
                "version": {"type": "string"},
                "revision": {"type": "string"},
                "vendor": {"type": "string"},
                "licenses": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "ModelFS": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["layers"]},
                "diffIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "additionalProperties": False,
            "required": ["type", "diffIds"],
        },
        "ModelCapabilities": {
            "type": "object",
            "properties": {
                "inputTypes": {"type": "array", "items": {"$ref": "#/$defs/Modality"}},
                "outputTypes": {"type": "array", "items": {"$ref": "#/$defs/Modality"}},
                "knowledgeCutoff": {"type": "string", "format": "date-time"},
                "reasoning": {"type": "boolean"},
                "toolUsage": {"type": "boolean"},
                "reward": {"type": "boolean"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^[a-z]{2}$"},
                },
            },
            "additionalProperties": False,
        },
        "Modality": {
            "type": "string",
            "enum": ["text", "image", "audio", "video", "embedding", "other"],
        },
    },
}
