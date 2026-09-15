from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import oras.client
import requests
import zstandard

from modelpack_client import (
    InvalidModelPackError,
    ModelCapabilities,
    ModelDescriptor,
    ModelLayer,
    ModelPackClient,
    ModelPackConfig,
    ModelTechnicalConfig,
    UnsafePathError,
)
from modelpack_client.__main__ import _parse_layer
from modelpack_client.constants import (
    CODE_RAW_MEDIA_TYPE,
    CODE_TAR_MEDIA_TYPE,
    DATASET_RAW_MEDIA_TYPE,
    DATASET_TAR_MEDIA_TYPE,
    DOC_RAW_MEDIA_TYPE,
    DOC_TAR_GZIP_MEDIA_TYPE,
    DOC_TAR_MEDIA_TYPE,
    FILE_METADATA_ANNOTATION,
    FILEPATH_ANNOTATION,
    MODEL_CONFIG_MEDIA_TYPE,
    MODEL_MANIFEST_ARTIFACT_TYPE,
    OCI_MANIFEST_MEDIA_TYPE,
    OCI_TITLE_ANNOTATION,
    WEIGHT_CONFIG_RAW_MEDIA_TYPE,
    WEIGHT_CONFIG_TAR_MEDIA_TYPE,
    WEIGHT_RAW_MEDIA_TYPE,
    WEIGHT_TAR_MEDIA_TYPE,
    WEIGHT_TAR_ZSTD_MEDIA_TYPE,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
REFERENCE = "registry.example/models/tiny:1"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _response(request, status, body=b"", headers=None):
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response.raw = BytesIO(body)
    response.url = request.url
    response.request = request
    return response


class FakeAuth:
    def __init__(self) -> None:
        self.calls = []

    def load_configs(self, container, configs=None) -> None:
        self.calls.append((container, configs))


class FakeOrasClient(oras.client.OrasClient):
    """In-memory registry behind oras's own blob, upload and manifest code."""

    ROUTE = re.compile(
        r"/v2/(?P<name>.+?)/(?P<kind>blobs/uploads|blobs|manifests)/(?P<ref>[^/]*)"
    )

    def __init__(self) -> None:
        super().__init__(hostname="registry.example")
        self.auth = FakeAuth()
        self.blobs: dict[str, bytes] = {}
        self.manifest_bytes: bytes | None = None
        self.requests: list[tuple[str, str]] = []
        self.request_headers: list[dict[str, str]] = []
        self._uploads: dict[str, bytearray] = {}

    @property
    def manifest(self):
        if self.manifest_bytes is None:
            return None
        return json.loads(self.manifest_bytes)

    @manifest.setter
    def manifest(self, value) -> None:
        self.manifest_bytes = json.dumps(value).encode()

    def do_request(self, url, method="GET", data=None, headers=None, json=None, stream=False):
        # Preparing the request like requests does also sizes file-backed bodies.
        request = requests.Request(method, url, data=data, headers=headers).prepare()
        path = urlsplit(url).path
        self.requests.append((method, path))
        self.request_headers.append(dict(headers or {}))
        match = self.ROUTE.fullmatch(path)
        if match is None:
            return _response(request, 404)
        kind, ref = match["kind"], match["ref"]

        if kind == "blobs":
            if ref not in self.blobs:
                return _response(request, 404)
            return _response(request, 200, self.blobs[ref] if method == "GET" else b"")

        if kind == "blobs/uploads":
            if method == "POST":
                upload = str(len(self._uploads))
                self._uploads[upload] = bytearray()
                location = f"/v2/{match['name']}/blobs/uploads/{upload}"
                return _response(request, 202, headers={"Location": location})
            body = request.body
            if hasattr(body, "read"):
                body = body.read()
            self._uploads[ref] += body or b""
            if method == "PATCH":
                return _response(request, 202, headers={"Location": path})
            content = bytes(self._uploads.pop(ref))
            digest = parse_qs(urlsplit(url).query)["digest"][0]
            if _sha256(content) != digest:
                return _response(request, 400)
            self.blobs[digest] = content
            return _response(request, 201, headers={"Docker-Content-Digest": digest})

        if method == "PUT":
            self.manifest_bytes = request.body
            return _response(
                request, 201, headers={"Docker-Content-Digest": _sha256(request.body)}
            )
        if self.manifest_bytes is None:
            return _response(request, 404)
        return _response(
            request,
            200,
            self.manifest_bytes,
            {
                "Content-Type": OCI_MANIFEST_MEDIA_TYPE,
                "Docker-Content-Digest": _sha256(self.manifest_bytes),
            },
        )


class ModelPackClientTest(unittest.TestCase):
    def test_push_builds_modelpack_manifest_and_pull_restores_files(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.safetensors"
            model.write_bytes(b"model weights")
            config = ModelPackConfig(
                descriptor=ModelDescriptor(name="tiny", version="1"),
                config=ModelTechnicalConfig(
                    architecture="transformer", format="safetensors"
                ),
            )

            pushed = client.push(
                REFERENCE,
                [ModelLayer(model, artifact_path="weights/model.safetensors")],
                config,
            )

            self.assertEqual(pushed.digest, _sha256(backend.manifest_bytes))
            self.assertEqual(
                backend.manifest["artifactType"], MODEL_MANIFEST_ARTIFACT_TYPE
            )
            self.assertEqual(
                backend.manifest["config"]["mediaType"], MODEL_CONFIG_MEDIA_TYPE
            )
            self.assertEqual(
                backend.manifest["layers"][0]["annotations"][FILEPATH_ANNOTATION],
                "weights/model.safetensors",
            )
            metadata = json.loads(
                backend.manifest["layers"][0]["annotations"][FILE_METADATA_ANNOTATION]
            )
            self.assertEqual(metadata["name"], "model.safetensors")
            self.assertEqual(metadata["size"], len(b"model weights"))

            output = root / "output"
            pulled = client.pull(REFERENCE, output)
            self.assertEqual(
                (output / "weights/model.safetensors").read_bytes(), b"model weights"
            )
            self.assertEqual(pulled.config["descriptor"]["name"], "tiny")
            self.assertEqual(len(pulled.config["modelfs"]["diffIds"]), 1)
            self.assertEqual(
                sorted(path.name for path in output.rglob("*")),
                ["model.safetensors", "weights"],
            )

    def test_directory_is_reproducibly_packed(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights"
            source.mkdir()
            (source / "b.bin").write_bytes(b"b")
            (source / "a.bin").write_bytes(b"a")

            client.push(
                REFERENCE,
                [ModelLayer(source, artifact_path="renamed/weights")],
                ModelPackConfig(),
            )
            layer = backend.manifest["layers"][0]
            archive_bytes = backend.blobs[layer["digest"]]
            archive = root / "layer.tar"
            archive.write_bytes(archive_bytes)
            with tarfile.open(archive, "r:") as tar:
                self.assertEqual(
                    tar.getnames(),
                    [
                        "renamed/weights",
                        "renamed/weights/a.bin",
                        "renamed/weights/b.bin",
                    ],
                )
                self.assertTrue(all(member.mtime == 0 for member in tar.getmembers()))

            output = root / "output"
            client.pull(REFERENCE, output)
            self.assertEqual((output / "renamed/weights/a.bin").read_bytes(), b"a")
            self.assertEqual((output / "renamed/weights/b.bin").read_bytes(), b"b")

    def test_directory_digest_does_not_depend_on_file_modes(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digests = []
            for name, data_mode, script_mode in (("one", 0o600, 0o700), ("two", 0o664, 0o775)):
                source = root / name / "weights"
                source.mkdir(parents=True)
                (source / "model.bin").write_bytes(b"weights")
                (source / "model.bin").chmod(data_mode)
                (source / "run.sh").write_bytes(b"#!/bin/sh\n")
                (source / "run.sh").chmod(script_mode)
                pushed = client.push(REFERENCE, [source], ModelPackConfig())
                digests.append(pushed.manifest["layers"][0]["digest"])

            self.assertEqual(digests[0], digests[1])
            with tarfile.open(fileobj=BytesIO(backend.blobs[digests[0]])) as tar:
                modes = {member.name: member.mode for member in tar.getmembers()}
            self.assertEqual(
                modes,
                {"weights": 0o755, "weights/model.bin": 0o644, "weights/run.sh": 0o755},
            )

    def test_push_stores_hard_links_as_regular_files(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights"
            source.mkdir()
            (source / "a.bin").write_bytes(b"shared")
            os.link(source / "a.bin", source / "b.bin")

            client.push(REFERENCE, [source], ModelPackConfig())
            output = root / "output"
            client.pull(REFERENCE, output)

            self.assertEqual((output / "weights/a.bin").read_bytes(), b"shared")
            self.assertEqual((output / "weights/b.bin").read_bytes(), b"shared")

    def test_symlinked_file_is_pushed_under_the_link_name(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blobs" / "3f2a9c"
            blob.parent.mkdir()
            blob.write_bytes(b"weights")
            link = root / "snapshots" / "rev" / "model.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to(Path("../../blobs/3f2a9c"))

            client.push(REFERENCE, [link], ModelPackConfig())

            annotations = backend.manifest["layers"][0]["annotations"]
            self.assertEqual(annotations[FILEPATH_ANNOTATION], "model.safetensors")
            self.assertEqual(annotations[OCI_TITLE_ANNOTATION], "model.safetensors")
            metadata = json.loads(annotations[FILE_METADATA_ANNOTATION])
            self.assertEqual(metadata["name"], "model.safetensors")
            self.assertEqual(metadata["size"], len(b"weights"))

    def test_file_metadata_uses_modctl_typeflags_and_permission_bits(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"weights")
            model.chmod(0o640)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_bytes(b"readme")

            client.push(
                REFERENCE,
                [model, ModelLayer(docs, media_type=DOC_RAW_MEDIA_TYPE)],
                ModelPackConfig(),
            )

            file_metadata, directory_metadata = (
                json.loads(layer["annotations"][FILE_METADATA_ANNOTATION])
                for layer in backend.manifest["layers"]
            )
            self.assertEqual(file_metadata["typeflag"], 0)
            self.assertEqual(file_metadata["mode"], 0o640)
            self.assertEqual(directory_metadata["typeflag"], 5)
            self.assertLessEqual(directory_metadata["mode"], 0o777)

    def test_directories_use_the_tar_type_of_their_raw_type(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        expected = {
            WEIGHT_RAW_MEDIA_TYPE: WEIGHT_TAR_MEDIA_TYPE,
            WEIGHT_CONFIG_RAW_MEDIA_TYPE: WEIGHT_CONFIG_TAR_MEDIA_TYPE,
            DOC_RAW_MEDIA_TYPE: DOC_TAR_MEDIA_TYPE,
            CODE_RAW_MEDIA_TYPE: CODE_TAR_MEDIA_TYPE,
            DATASET_RAW_MEDIA_TYPE: DATASET_TAR_MEDIA_TYPE,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "files"
            source.mkdir()
            (source / "file.txt").write_bytes(b"contents")
            for raw_type, tar_type in expected.items():
                with self.subTest(raw_type):
                    pushed = client.push(
                        REFERENCE,
                        [ModelLayer(source, media_type=raw_type)],
                        ModelPackConfig(),
                    )
                    self.assertEqual(pushed.manifest["layers"][0]["mediaType"], tar_type)

    def test_compressed_directory_layers_round_trip(self):
        decompress = {
            DOC_TAR_GZIP_MEDIA_TYPE: gzip.decompress,
            WEIGHT_TAR_ZSTD_MEDIA_TYPE: lambda data: zstandard.ZstdDecompressor()
            .stream_reader(BytesIO(data))
            .read(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "files"
            source.mkdir()
            (source / "file.txt").write_bytes(b"contents" * 1000)
            for media_type, decompressor in decompress.items():
                with self.subTest(media_type):
                    backend = FakeOrasClient()
                    client = ModelPackClient(oras_client=backend)
                    layer = ModelLayer(source, media_type=media_type)
                    first = client.push(REFERENCE, [layer], ModelPackConfig())
                    second = client.push(REFERENCE, [layer], ModelPackConfig())
                    self.assertEqual(first.digest, second.digest)

                    descriptor = first.manifest["layers"][0]
                    self.assertEqual(descriptor["mediaType"], media_type)
                    uncompressed = decompressor(backend.blobs[descriptor["digest"]])
                    output = root / media_type.rpartition("+")[2]
                    pulled = client.pull(REFERENCE, output)
                    self.assertEqual(
                        pulled.config["modelfs"]["diffIds"], [_sha256(uncompressed)]
                    )
                    self.assertEqual(
                        (output / "files/file.txt").read_bytes(), b"contents" * 1000
                    )

    def test_push_streams_blobs_in_one_put_unless_chunked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"0123456789" * 10)
            for chunked in (False, True):
                with self.subTest(chunked=chunked):
                    backend = FakeOrasClient()
                    client = ModelPackClient(oras_client=backend)
                    client.push(
                        REFERENCE, [model], ModelPackConfig(), chunked=chunked, chunk_size=16
                    )
                    methods = {method for method, _ in backend.requests}
                    self.assertEqual("PATCH" in methods, chunked)
                    output = root / f"output-{chunked}"
                    client.pull(REFERENCE, output)
                    self.assertEqual((output / "model.bin").read_bytes(), model.read_bytes())

    def test_pull_rejects_non_modelpack_artifact(self):
        backend = FakeOrasClient()
        backend.manifest = {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "config": {"mediaType": MODEL_CONFIG_MEDIA_TYPE},
            "layers": [],
        }
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidModelPackError):
                client.pull("registry.example/not-a-model:1", directory)

    def test_pull_rejects_path_traversal(self):
        backend = FakeOrasClient()
        config = {
            "descriptor": {},
            "config": {},
            "modelfs": {"type": "layers", "diffIds": ["sha256:" + "0" * 64]},
        }
        config_bytes = json.dumps(config).encode()
        config_digest = _sha256(config_bytes)
        backend.blobs[config_digest] = config_bytes
        backend.manifest = {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": MODEL_MANIFEST_ARTIFACT_TYPE,
            "config": {
                "mediaType": MODEL_CONFIG_MEDIA_TYPE,
                "digest": config_digest,
                "size": len(config_bytes),
            },
            "layers": [
                {
                    "mediaType": WEIGHT_RAW_MEDIA_TYPE,
                    "digest": "sha256:" + "1" * 64,
                    "size": 1,
                    "annotations": {FILEPATH_ANNOTATION: "../escape"},
                }
            ],
        }
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UnsafePathError):
                client.pull("registry.example/models/unsafe:1", directory)

    def test_pull_rejects_blob_with_wrong_digest(self):
        backend = FakeOrasClient()
        config = {
            "descriptor": {},
            "config": {},
            "modelfs": {"type": "layers", "diffIds": []},
        }
        config_bytes = json.dumps(config).encode()
        config_digest = "sha256:" + "0" * 64
        backend.blobs[config_digest] = config_bytes
        backend.manifest = {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": MODEL_MANIFEST_ARTIFACT_TYPE,
            "config": {
                "mediaType": MODEL_CONFIG_MEDIA_TYPE,
                "digest": config_digest,
                "size": len(config_bytes),
            },
            "layers": [],
        }
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InvalidModelPackError, "digest mismatch"):
                client.pull("registry.example/models/corrupt:1", directory)

    def _backend_with_layers(self, backend, layers, *, config=None):
        """Store a manifest whose layers are given as (bytes, media type, annotations)."""
        descriptors = []
        diff_ids = []
        for layer_bytes, media_type, layer_annotations, *diff_id in layers:
            digest = _sha256(layer_bytes)
            backend.blobs[digest] = layer_bytes
            diff_ids.append(diff_id[0] if diff_id else digest)
            descriptors.append(
                {
                    "mediaType": media_type,
                    "digest": digest,
                    "size": len(layer_bytes),
                    "annotations": layer_annotations,
                }
            )
        if config is None:
            config = {
                "descriptor": {},
                "config": {},
                "modelfs": {"type": "layers", "diffIds": diff_ids},
            }
        config_bytes = json.dumps(config).encode()
        config_digest = _sha256(config_bytes)
        backend.blobs[config_digest] = config_bytes
        backend.manifest = {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": MODEL_MANIFEST_ARTIFACT_TYPE,
            "config": {
                "mediaType": MODEL_CONFIG_MEDIA_TYPE,
                "digest": config_digest,
                "size": len(config_bytes),
            },
            "layers": descriptors,
        }
        return backend

    def _backend_with_layer(
        self,
        backend,
        layer_bytes,
        media_type,
        annotations,
        *,
        diff_id=None,
    ):
        layer = (layer_bytes, media_type, annotations)
        if diff_id is not None:
            layer += (diff_id,)
        return self._backend_with_layers(backend, [layer])

    def test_pull_accepts_tar_root_directory_entry(self):
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.tar"
            payload = root / "payload"
            payload.mkdir()
            (payload / "weights.bin").write_bytes(b"weights")
            with tarfile.open(archive, "w") as tar:
                tar.add(payload, arcname=".")
            self._backend_with_layer(
                backend,
                archive.read_bytes(),
                WEIGHT_TAR_MEDIA_TYPE,
                {FILEPATH_ANNOTATION: "payload.tar"},
            )

            output = root / "out"
            result = ModelPackClient(oras_client=backend).pull(REFERENCE, output)

            self.assertEqual((output / "weights.bin").read_bytes(), b"weights")
            self.assertEqual(result.files, ((output / "weights.bin").resolve(),))

    def test_pull_tar_layer_does_not_duplicate_filepath_prefix(self):
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.tar"
            payload = root / "a.bin"
            payload.write_bytes(b"weights")
            with tarfile.open(archive, "w") as tar:
                tar.add(payload, arcname="weights/a.bin")
            self._backend_with_layer(
                backend,
                archive.read_bytes(),
                WEIGHT_TAR_MEDIA_TYPE,
                {FILEPATH_ANNOTATION: "weights"},
            )

            output = root / "out"
            ModelPackClient(oras_client=backend).pull(REFERENCE, output)

            self.assertEqual((output / "weights/a.bin").read_bytes(), b"weights")
            self.assertFalse((output / "weights/weights").exists())

    def test_read_only_tar_directory_does_not_block_later_layers(self):
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                member = tarfile.TarInfo("weights")
                member.type = tarfile.DIRTYPE
                member.mode = 0o555
                tar.addfile(member)
                member = tarfile.TarInfo("weights/a.bin")
                member.size = 1
                tar.addfile(member, BytesIO(b"a"))
            self._backend_with_layers(
                backend,
                [
                    (archive.getvalue(), WEIGHT_TAR_MEDIA_TYPE, {FILEPATH_ANNOTATION: "weights"}),
                    (b"b", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "weights/b.bin"}),
                ],
            )
            output = root / "out"
            (output / "weights").mkdir(parents=True)
            (output / "weights").chmod(0o750)
            before = (output / "weights").stat()

            ModelPackClient(oras_client=backend).pull(REFERENCE, output)

            self.assertEqual((output / "weights/a.bin").read_bytes(), b"a")
            self.assertEqual((output / "weights/b.bin").read_bytes(), b"b")
            after = (output / "weights").stat()
            self.assertEqual(stat.S_IMODE(after.st_mode), 0o750)
            self.assertGreaterEqual(after.st_mtime, before.st_mtime)

    def test_pull_rejects_layer_with_wrong_diff_id(self):
        backend = FakeOrasClient()
        self._backend_with_layer(
            backend,
            b"weights",
            WEIGHT_RAW_MEDIA_TYPE,
            {FILEPATH_ANNOTATION: "weights.bin"},
            diff_id="sha256:" + "0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(InvalidModelPackError, "DiffID mismatch"):
                ModelPackClient(oras_client=backend).pull(
                    "registry.example/models/corrupt:1", output
                )
            self.assertFalse((output / "weights.bin").exists())

    def test_pull_rejects_compressed_layer_with_wrong_diff_id_before_writing_files(self):
        backend = FakeOrasClient()
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            member = tarfile.TarInfo("weights/model.bin")
            member.size = 7
            tar.addfile(member, BytesIO(b"weights"))
        self._backend_with_layer(
            backend,
            gzip.compress(archive.getvalue()),
            "application/vnd.cncf.model.weight.v1.tar+gzip",
            {FILEPATH_ANNOTATION: "weights"},
            diff_id="sha256:" + "0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(InvalidModelPackError, "DiffID mismatch"):
                ModelPackClient(oras_client=backend).pull(REFERENCE, output)
            self.assertEqual(
                [path for path in output.rglob("*") if path.is_file()], []
            )

    def test_pull_streams_zstd_tar_without_uncompressed_temporary_file(self):
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_buffer = BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w") as tar:
                payload = b"weights"
                member = tarfile.TarInfo("weights/model.bin")
                member.size = len(payload)
                tar.addfile(member, BytesIO(payload))
            uncompressed = archive_buffer.getvalue()
            compressed = zstandard.ZstdCompressor().compress(uncompressed)
            self._backend_with_layer(
                backend,
                compressed,
                WEIGHT_TAR_ZSTD_MEDIA_TYPE,
                {FILEPATH_ANNOTATION: "weights"},
                diff_id=_sha256(uncompressed),
            )

            output = root / "out"
            ModelPackClient(oras_client=backend).pull(REFERENCE, output)

            self.assertEqual((output / "weights/model.bin").read_bytes(), b"weights")
            self.assertFalse(any(output.glob("*.tar")))

    def test_pull_rejects_dot_filepath_annotation(self):
        """A '.' filepath must not resolve to the parent of the destination."""
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "payload.tar"
            source_dir = root / "src"
            source_dir.mkdir()
            member = source_dir / "PWNED.txt"
            member.write_bytes(b"pwned")
            with tarfile.open(archive, "w") as tar:
                tar.add(member, arcname="PWNED.txt")
            self._backend_with_layer(
                backend,
                archive.read_bytes(),
                WEIGHT_TAR_MEDIA_TYPE,
                {FILEPATH_ANNOTATION: "."},
            )
            client = ModelPackClient(oras_client=backend)
            output = root / "outdir"
            with self.assertRaises(UnsafePathError):
                client.pull("registry.example/models/evil:1", output)
            self.assertFalse((root / "PWNED.txt").exists())

    def test_pull_strips_setuid_from_file_metadata(self):
        backend = FakeOrasClient()
        metadata = json.dumps({"name": "weights.bin", "mode": 0o4755})
        self._backend_with_layer(
            backend,
            b"weights",
            WEIGHT_RAW_MEDIA_TYPE,
            {FILEPATH_ANNOTATION: "weights.bin", FILE_METADATA_ANNOTATION: metadata},
        )
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            result = client.pull("registry.example/models/setuid:1", directory)
            mode = result.files[0].stat().st_mode
            self.assertFalse(mode & stat.S_ISUID)
            self.assertFalse(mode & stat.S_ISGID)
            self.assertEqual(stat.S_IMODE(mode), 0o755)

    def test_pulled_file_stays_readable_and_writable_by_owner(self):
        backend = FakeOrasClient()
        metadata = json.dumps({"name": "weights.bin", "mode": 0})
        self._backend_with_layer(
            backend,
            b"weights",
            WEIGHT_RAW_MEDIA_TYPE,
            {FILEPATH_ANNOTATION: "weights.bin", FILE_METADATA_ANNOTATION: metadata},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ModelPackClient(oras_client=backend).pull(REFERENCE, directory)
            self.assertEqual(stat.S_IMODE(result.files[0].stat().st_mode), 0o600)
            self.assertEqual(result.files[0].read_bytes(), b"weights")

    def test_pull_applies_go_nanosecond_mtime(self):
        backend = FakeOrasClient()
        metadata = json.dumps(
            {
                "name": "weights.bin",
                "mode": 0o644,
                "mtime": "2025-03-10T15:04:05.123456789+09:00",
            }
        )
        self._backend_with_layer(
            backend,
            b"weights",
            WEIGHT_RAW_MEDIA_TYPE,
            {FILEPATH_ANNOTATION: "weights.bin", FILE_METADATA_ANNOTATION: metadata},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ModelPackClient(oras_client=backend).pull(REFERENCE, directory)
            expected = datetime(2025, 3, 10, 6, 4, 5, 123456, tzinfo=timezone.utc)
            self.assertAlmostEqual(
                result.files[0].stat().st_mtime, expected.timestamp(), places=5
            )

    def test_invalid_file_metadata_leaves_no_file_behind(self):
        backend = FakeOrasClient()
        metadata = json.dumps({"name": "weights.bin", "mtime": "garbage"})
        self._backend_with_layer(
            backend,
            b"weights",
            WEIGHT_RAW_MEDIA_TYPE,
            {FILEPATH_ANNOTATION: "weights.bin", FILE_METADATA_ANNOTATION: metadata},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InvalidModelPackError, "file metadata"):
                ModelPackClient(oras_client=backend).pull(REFERENCE, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pull_by_digest_verifies_manifest(self):
        backend = FakeOrasClient()
        self._backend_with_layer(
            backend, b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "w.bin"}
        )
        client = ModelPackClient(oras_client=backend)
        manifest_bytes = backend.manifest_bytes
        digests = {
            "sha256": _sha256(manifest_bytes),
            "sha512": "sha512:" + hashlib.sha512(manifest_bytes).hexdigest(),
        }
        for algorithm, digest in digests.items():
            with self.subTest(algorithm), tempfile.TemporaryDirectory() as directory:
                reference = f"registry.example/models/tiny@{digest}"
                backend.manifest_bytes = manifest_bytes
                client.pull(reference, directory)

                backend.manifest_bytes = manifest_bytes + b" "
                with self.assertRaisesRegex(InvalidModelPackError, "manifest digest"):
                    client.pull(reference, directory)

    def test_pull_rejects_malformed_manifest_and_config(self):
        cases = {
            "layer without digest": lambda manifest: manifest["layers"][0].pop("digest"),
            "layer annotations as a list": lambda manifest: manifest["layers"][0].update(
                annotations=[FILEPATH_ANNOTATION]
            ),
            "config without size": lambda manifest: manifest["config"].pop("size"),
            "annotation that is not a string": lambda manifest: manifest["layers"][0][
                "annotations"
            ].update({FILEPATH_ANNOTATION: 1}),
        }
        for name, corrupt in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                backend = self._backend_with_layer(
                    FakeOrasClient(),
                    b"weights",
                    WEIGHT_RAW_MEDIA_TYPE,
                    {FILEPATH_ANNOTATION: "w.bin"},
                )
                manifest = backend.manifest
                corrupt(manifest)
                backend.manifest = manifest
                with self.assertRaises(InvalidModelPackError):
                    ModelPackClient(oras_client=backend).pull(REFERENCE, directory)

        with tempfile.TemporaryDirectory() as directory:
            backend = self._backend_with_layers(
                FakeOrasClient(),
                [(b"weights", WEIGHT_RAW_MEDIA_TYPE, {})],
                config=["not", "an", "object"],
            )
            with self.assertRaises(InvalidModelPackError):
                ModelPackClient(oras_client=backend).pull(REFERENCE, directory)

    def test_push_rejects_config_that_does_not_match_the_schema(self):
        cases = {
            "no layers": ([], ModelPackConfig()),
            "unknown descriptor field": (
                None,
                {"descriptor": {"created_at": "2025-01-01T00:00:00Z"}, "config": {}},
            ),
            "language that is not an ISO 639-1 code": (
                None,
                ModelPackConfig(
                    config=ModelTechnicalConfig(
                        capabilities=ModelCapabilities(languages=("english",))
                    )
                ),
            ),
            "createdAt without a time zone": (
                None,
                ModelPackConfig(descriptor=ModelDescriptor(created_at="2025-01-01T00:00:00")),
            ),
            "empty name": (None, ModelPackConfig(descriptor=ModelDescriptor(name=""))),
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.bin"
            model.write_bytes(b"weights")
            for name, (layers, config) in cases.items():
                with self.subTest(name):
                    backend = FakeOrasClient()
                    with self.assertRaises(InvalidModelPackError):
                        ModelPackClient(oras_client=backend).push(
                            REFERENCE, [model] if layers is None else layers, config
                        )
                    self.assertEqual(backend.blobs, {})

            ModelPackClient(oras_client=FakeOrasClient()).push(
                REFERENCE,
                [model],
                ModelPackConfig(
                    descriptor=ModelDescriptor(
                        created_at="2025-03-10T15:04:05.123456789+09:00"
                    )
                ),
            )

    def test_pull_with_keep_does_not_trip_over_previous_pulls(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"weights")
            client.push(REFERENCE, [model], ModelPackConfig())
            output = root / "output"

            client.pull(REFERENCE, output, overwrite=False)
            self.assertEqual([path.name for path in output.iterdir()], ["model.bin"])
            (output / "model.bin").unlink()
            client.pull(REFERENCE, output, overwrite=False)

            self.assertEqual((output / "model.bin").read_bytes(), b"weights")

    def test_pull_without_unpack_saves_tar_layers_as_archives(self):
        backend = FakeOrasClient()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights"
            source.mkdir()
            (source / "a.bin").write_bytes(b"a")
            client.push(REFERENCE, [source], ModelPackConfig())
            output = root / "output"

            client.pull(REFERENCE, output)
            result = client.pull(REFERENCE, output, unpack=False)

            self.assertEqual(result.files, ((output / "weights.tar").resolve(),))
            with tarfile.open(output / "weights.tar") as tar:
                self.assertEqual(tar.getnames(), ["weights", "weights/a.bin"])
            self.assertEqual((output / "weights/a.bin").read_bytes(), b"a")

    def test_raw_layer_does_not_replace_a_directory(self):
        backend = FakeOrasClient()
        self._backend_with_layer(
            backend, b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "weights"}
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "weights").mkdir()
            (output / "weights/keep.bin").write_bytes(b"keep")
            with self.assertRaisesRegex(IsADirectoryError, "cannot replace a directory"):
                ModelPackClient(oras_client=backend).pull(REFERENCE, output)
            self.assertEqual((output / "weights/keep.bin").read_bytes(), b"keep")

    def test_push_rejects_layers_that_would_be_pulled_to_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root / name).mkdir()
                (root / name / "config.json").write_bytes(name.encode())
            backend = FakeOrasClient()
            client = ModelPackClient(oras_client=backend)

            with self.assertRaisesRegex(InvalidModelPackError, "config.json"):
                client.push(
                    REFERENCE, [root / "a/config.json", root / "b/config.json"], ModelPackConfig()
                )
            self.assertEqual(backend.blobs, {})

            client.push(
                REFERENCE,
                [
                    ModelLayer(root / "a/config.json", artifact_path="a/config.json"),
                    ModelLayer(root / "b/config.json", artifact_path="b/config.json"),
                ],
                ModelPackConfig(),
            )

    def test_corrupt_gzip_layer_is_invalid(self):
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            member = tarfile.TarInfo("weights/model.bin")
            member.size = 1_000_000
            tar.addfile(member, BytesIO(os.urandom(member.size)))
        # Break the gzip stream with a deflate block of the reserved type, far enough
        # into the file contents that tarfile reads the header without trouble.
        compressor = zlib.compressobj(wbits=31)
        corrupt = (
            compressor.compress(archive.getvalue()[:600_000])
            + compressor.flush(zlib.Z_FULL_FLUSH)
            + b"\x07"
        )
        media_type = "application/vnd.cncf.model.weight.v1.tar+gzip"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layer = root / "weights.tar.gz"
            layer.write_bytes(corrupt)
            with self.subTest("push"), self.assertRaises(InvalidModelPackError):
                ModelPackClient(oras_client=FakeOrasClient()).push(
                    REFERENCE, [ModelLayer(layer, media_type=media_type)], ModelPackConfig()
                )

            backend = self._backend_with_layer(
                FakeOrasClient(),
                bytes(corrupt),
                media_type,
                {FILEPATH_ANNOTATION: "weights"},
                diff_id="sha256:" + "0" * 64,
            )
            with self.subTest("pull"), self.assertRaises(InvalidModelPackError):
                ModelPackClient(oras_client=backend).pull(REFERENCE, root / "out")

    def test_pull_rejects_tar_layer_with_a_member_twice(self):
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name, data in (("x/one", b"one"), ("x/dup", b"first"), ("x/dup", b"second")):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, BytesIO(data))
        backend = self._backend_with_layer(
            FakeOrasClient(), archive.getvalue(), WEIGHT_TAR_MEDIA_TYPE, {FILEPATH_ANNOTATION: "x"}
        )
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with self.assertRaisesRegex(InvalidModelPackError, "more than once"):
                    ModelPackClient(oras_client=backend).pull(
                        REFERENCE, output, overwrite=overwrite
                    )
                self.assertEqual([path for path in output.rglob("*") if path.is_file()], [])

    def test_pull_stops_reading_a_blob_longer_than_its_descriptor(self):
        backend = self._backend_with_layer(
            FakeOrasClient(), b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "w.bin"}
        )
        digest = backend.manifest["layers"][0]["digest"]
        backend.blobs[digest] = b"weights" + b"\0" * (4 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(InvalidModelPackError, "longer than its descriptor"):
                ModelPackClient(oras_client=backend).pull(REFERENCE, output)
            self.assertEqual(list(output.iterdir()), [])

    def test_pull_rejects_manifest_over_4_mib(self):
        backend = self._backend_with_layer(
            FakeOrasClient(), b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "w.bin"}
        )
        backend.manifest_bytes += b" " * (4 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InvalidModelPackError, "manifest is larger"):
                ModelPackClient(oras_client=backend).pull(REFERENCE, directory)

    def test_requests_carry_the_oras_client_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"
            model.write_bytes(b"weights")
            for chunked in (False, True):
                with self.subTest(chunked=chunked):
                    backend = FakeOrasClient()
                    backend.set_header("X-Registry-Key", "key")
                    client = ModelPackClient(oras_client=backend)
                    client.push(REFERENCE, [model], ModelPackConfig(), chunked=chunked)
                    client.pull(REFERENCE, root / f"out-{chunked}")

                    self.assertIn("POST", {method for method, _ in backend.requests})
                    for (method, path), headers in zip(backend.requests, backend.request_headers, strict=True):
                        self.assertEqual(headers.get("X-Registry-Key"), "key", (method, path))

    def test_pulled_file_without_mode_follows_the_umask(self):
        backend = self._backend_with_layer(
            FakeOrasClient(), b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "w.bin"}
        )
        umask = os.umask(0o027)
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = ModelPackClient(oras_client=backend).pull(REFERENCE, directory)
                self.assertEqual(stat.S_IMODE(result.files[0].stat().st_mode), 0o640)
        finally:
            os.umask(umask)

    def test_empty_file_layer_when_the_registry_refuses_the_empty_blob(self):
        class RefusingEmptyBlob(FakeOrasClient):
            EMPTY = _sha256(b"")

            def do_request(self, url, method="GET", data=None, headers=None, json=None, stream=False):
                parts = urlsplit(url)
                if self.EMPTY in parts.path or [self.EMPTY] == parse_qs(parts.query).get("digest"):
                    request = requests.Request(method, url).prepare()
                    return _response(request, 404 if method in ("GET", "HEAD") else 400)
                return super().do_request(url, method, data, headers, json, stream)

        backend = RefusingEmptyBlob()
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "__init__.py"
            empty.write_bytes(b"")
            client.push(REFERENCE, [empty], ModelPackConfig())
            self.assertNotIn(_sha256(b""), backend.blobs)

            result = client.pull(REFERENCE, root / "out")
            self.assertEqual(result.files[0].read_bytes(), b"")

    def test_pull_tar_layer_respects_overwrite_false(self):
        backend = FakeOrasClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "a.bin"
            payload.write_bytes(b"new")
            archive = root / "payload.tar"
            with tarfile.open(archive, "w") as tar:
                tar.add(payload, arcname="weights/a.bin")
            self._backend_with_layer(
                backend,
                archive.read_bytes(),
                WEIGHT_TAR_MEDIA_TYPE,
                {FILEPATH_ANNOTATION: "weights"},
            )
            client = ModelPackClient(oras_client=backend)
            output = root / "out"
            (output / "weights").mkdir(parents=True)
            (output / "weights" / "a.bin").write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                client.pull(REFERENCE, output, overwrite=False)
            self.assertEqual((output / "weights" / "a.bin").read_bytes(), b"old")


class CommandLineTest(unittest.TestCase):
    def test_python_m_modelpack_runs_the_cli(self):
        result = subprocess.run(
            [sys.executable, "-m", "modelpack_client", "--help"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: modelpack", result.stdout)

    def test_layer_paths_may_contain_colons(self):
        layer = _parse_layer("models/v1:final.bin")
        self.assertEqual((layer.path, layer.media_type), ("models/v1:final.bin", WEIGHT_RAW_MEDIA_TYPE))

        layer = _parse_layer(f"docs:v1/README.md:{DOC_RAW_MEDIA_TYPE}")
        self.assertEqual((layer.path, layer.media_type), ("docs:v1/README.md", DOC_RAW_MEDIA_TYPE))


if __name__ == "__main__":
    unittest.main()
