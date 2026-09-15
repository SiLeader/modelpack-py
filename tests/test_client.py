from __future__ import annotations

import hashlib
import json
import stat
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from modelpack import (
    InvalidModelPackError,
    ModelDescriptor,
    ModelLayer,
    ModelPackClient,
    ModelPackConfig,
    ModelTechnicalConfig,
    UnsafePathError,
)
from modelpack.constants import (
    FILEPATH_ANNOTATION,
    FILE_METADATA_ANNOTATION,
    MODEL_CONFIG_MEDIA_TYPE,
    MODEL_MANIFEST_ARTIFACT_TYPE,
    OCI_MANIFEST_MEDIA_TYPE,
    WEIGHT_RAW_MEDIA_TYPE,
    WEIGHT_TAR_MEDIA_TYPE,
)


class FakeAuth:
    def __init__(self) -> None:
        self.calls = []

    def load_configs(self, container, configs=None) -> None:
        self.calls.append((container, configs))


class FakeOrasClient:
    def __init__(self) -> None:
        self.auth = FakeAuth()
        self.blobs: dict[str, bytes] = {}
        self.manifest = None

    def get_container(self, reference):
        return SimpleNamespace(__str__=lambda self: reference, reference=reference)

    def upload_blob(self, path, container, descriptor, **kwargs):
        self.blobs[descriptor["digest"]] = Path(path).read_bytes()
        return SimpleNamespace(headers={}, status_code=201)

    def upload_manifest(self, manifest, container):
        self.manifest = manifest
        digest = "sha256:" + hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        return SimpleNamespace(
            headers={"Docker-Content-Digest": digest}, status_code=201
        )

    def _check_200_response(self, response):
        if response.status_code not in (200, 201):
            raise RuntimeError("request failed")

    def get_manifest(self, container, allowed_media_type=None):
        return self.manifest

    def download_blob(self, container, digest, destination):
        Path(destination).write_bytes(self.blobs[digest])


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
                "registry.example/models/tiny:1",
                [ModelLayer(model, artifact_path="weights/model.safetensors")],
                config,
            )

            self.assertIsNotNone(pushed.digest)
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
            pulled = client.pull("registry.example/models/tiny:1", output)
            self.assertEqual(
                (output / "weights/model.safetensors").read_bytes(), b"model weights"
            )
            self.assertEqual(pulled.config["descriptor"]["name"], "tiny")
            self.assertEqual(len(pulled.config["modelfs"]["diffIds"]), 1)

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
                "registry.example/models/tiny:1",
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
            client.pull("registry.example/models/tiny:1", output)
            self.assertEqual((output / "renamed/weights/a.bin").read_bytes(), b"a")
            self.assertEqual((output / "renamed/weights/b.bin").read_bytes(), b"b")

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
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
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


    def _backend_with_layer(
        self,
        backend,
        layer_bytes,
        media_type,
        annotations,
        *,
        diff_id=None,
    ):
        digest = "sha256:" + hashlib.sha256(layer_bytes).hexdigest()
        backend.blobs[digest] = layer_bytes
        if diff_id is None:
            diff_id = digest
        config = {
            "descriptor": {},
            "config": {},
            "modelfs": {"type": "layers", "diffIds": [diff_id]},
        }
        config_bytes = json.dumps(config).encode()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
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
                    "mediaType": media_type,
                    "digest": digest,
                    "size": len(layer_bytes),
                    "annotations": annotations,
                }
            ],
        }
        return backend

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
            result = ModelPackClient(oras_client=backend).pull(
                "registry.example/models/tiny:1", output
            )

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
            ModelPackClient(oras_client=backend).pull(
                "registry.example/models/tiny:1", output
            )

            self.assertEqual((output / "weights/a.bin").read_bytes(), b"weights")
            self.assertFalse((output / "weights/weights").exists())

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

    def test_pull_streams_zstd_tar_without_uncompressed_temporary_file(self):
        import zstandard

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
                "application/vnd.cncf.model.weight.v1.tar+zstd",
                {FILEPATH_ANNOTATION: "weights"},
                diff_id="sha256:" + hashlib.sha256(uncompressed).hexdigest(),
            )

            output = root / "out"
            ModelPackClient(oras_client=backend).pull(
                "registry.example/models/tiny:1", output
            )

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

    def test_pull_by_digest_requires_verifiable_manifest(self):
        backend = FakeOrasClient()
        self._backend_with_layer(
            backend, b"weights", WEIGHT_RAW_MEDIA_TYPE, {FILEPATH_ANNOTATION: "w.bin"}
        )
        client = ModelPackClient(oras_client=backend)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InvalidModelPackError, "digest reference"):
                client.pull(
                    "registry.example/models/tiny@sha256:" + "0" * 64, directory
                )

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
                client.pull(
                    "registry.example/models/tiny:1", output, overwrite=False
                )
            self.assertEqual((output / "weights" / "a.bin").read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
