# modelpack

Python client for pushing and pulling
[CNCF ModelPack](https://github.com/modelpack/model-spec) artifacts through any
OCI Distribution-compatible registry, including Harbor, Docker Hub, and GHCR.

## Installation

```console
pip install modelpack
```

## Python API

```python
from modelpack import (
    ModelDescriptor,
    ModelLayer,
    ModelPackClient,
    ModelPackConfig,
    ModelTechnicalConfig,
)

client = ModelPackClient()
client.login(
    "ghcr.io",
    username="octocat",
    password="TOKEN",
)

config = ModelPackConfig(
    descriptor=ModelDescriptor(
        name="example-model",
        version="1.0.0",
        licenses=("Apache-2.0",),
    ),
    config=ModelTechnicalConfig(
        architecture="transformer",
        format="safetensors",
        param_size="7B",
        precision="float16",
    ),
)

result = client.push(
    "ghcr.io/example/models/example:1.0.0",
    layers=[
        ModelLayer("model.safetensors"),
        ModelLayer(
            "tokenizer.json",
            media_type="application/vnd.cncf.model.weight.config.v1.raw",
        ),
    ],
    config=config,
)
print(result.digest)

pulled = client.pull(
    "ghcr.io/example/models/example:1.0.0",
    "./example-model",
)
print(pulled.files)
print(pulled.config)
```

Directories passed as layers are reproducibly packed as uncompressed tar
layers. The client sets the required ModelPack artifact/config media types,
generates `modelfs.diffIds`, validates pulled manifests, and rejects unsafe
artifact and archive paths.

To use Docker's existing credential file, omit `login()`. `oras-py` loads
credentials from the standard registry configuration. A custom file can be
passed as `config_path` to `push()` or `pull()`.

## CLI

```console
modelpack push ghcr.io/example/models/example:1.0.0 \
  --config model-config.json \
  --layer model.safetensors \
  --layer 'tokenizer.json:application/vnd.cncf.model.weight.config.v1.raw'

modelpack pull ghcr.io/example/models/example:1.0.0 ./model
```
