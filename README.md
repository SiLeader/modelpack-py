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

Directories passed as layers are packed as tar layers: a raw media type becomes
the matching tar type, and the `+gzip` and `+zstd` tar types are compressed.
Entries are sorted, and owners, timestamps and permissions (except the
executable bit) are normalized, so the same tree gets the same digest on any
machine. Hard links are stored as regular files; symbolic links are rejected.

The client sets the required ModelPack artifact/config media types, generates
`modelfs.diffIds`, validates configs against the specification's JSON schema,
verifies manifests and blobs against their digests, and rejects unsafe artifact
and archive paths.

`pull()` unpacks tar layers into the destination directory. With
`unpack=False`, they are saved as archives named after the layer's file path,
such as `weights.tar`. `push()` streams each blob in a single request; pass
`chunked=True` for registries that require chunked uploads. Each layer must be
pulled to its own path, so give files that share a name an `artifact_path`.

To use Docker's existing credential file, omit `login()`. `oras-py` loads
credentials from the standard registry configuration. A custom file can be
passed as `config_path` to `push()` or `pull()`, and its entries take
precedence. Credential files are read again on every call. `login()` saves
credentials to Docker's file or `config_path`, and `logout()` removes them from
the same file. Credentials and tokens are only sent to the registry they belong
to; an upload that the registry sends to another host goes without them.

## CLI

```console
modelpack push ghcr.io/example/models/example:1.0.0 \
  --config model-config.json \
  --layer model.safetensors \
  --layer 'tokenizer.json:application/vnd.cncf.model.weight.config.v1.raw'

modelpack pull ghcr.io/example/models/example:1.0.0 ./model
```

`python -m modelpack` runs the same CLI.
