"""CNCF ModelPack client backed by an OCI registry."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import oras.client
import oras.oci

from ..constants import (
    FILEPATH_ANNOTATION,
    FILE_METADATA_ANNOTATION,
    MODEL_CONFIG_MEDIA_TYPE,
    MODEL_LAYER_MEDIA_TYPES,
    MODEL_MANIFEST_ARTIFACT_TYPE,
    OCI_MANIFEST_MEDIA_TYPE,
    OCI_TITLE_ANNOTATION,
    TAR_MEDIA_TYPES,
    WEIGHT_RAW_MEDIA_TYPE,
    WEIGHT_TAR_MEDIA_TYPE,
)
from ..errors import InvalidModelPackError, UnsafePathError
from ..models import ModelLayer, ModelPackConfig, PullResult, PushResult

#: Blobs larger than this are uploaded with the chunked (streaming) flow so that
#: multi-gigabyte model weights are never buffered in memory.
CHUNKED_UPLOAD_THRESHOLD = 64 * 1024 * 1024


def _reference_digest(reference: str) -> str | None:
    """Return the digest of a ``repo@sha256:...`` reference, if present."""
    repository, separator, digest = reference.rpartition("@")
    if not separator or not digest.startswith("sha256:") or not repository:
        return None
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _file_metadata(path: Path) -> str:
    stat = path.stat()
    return json.dumps(
        {
            "name": path.name,
            "mode": stat.st_mode & 0o7777,
            "uid": stat.st_uid,
            "gid": stat.st_gid,
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "typeflag": ord("5") if path.is_dir() else 0,
        },
        separators=(",", ":"),
    )


def _apply_file_metadata(path: Path, annotation: str | None) -> None:
    if not annotation:
        return
    try:
        metadata = json.loads(annotation)
        mode = metadata.get("mode")
        mtime = metadata.get("mtime")
        if isinstance(mode, int):
            # Never honour setuid/setgid/sticky bits from an untrusted registry.
            path.chmod(mode & 0o777)
        if isinstance(mtime, str):
            timestamp = datetime.fromisoformat(mtime.replace("Z", "+00:00")).timestamp()
            os.utime(path, (timestamp, timestamp))
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        raise InvalidModelPackError("invalid ModelPack file metadata annotation") from error


def _safe_destination(root: Path, artifact_path: str) -> Path:
    relative = PurePosixPath(artifact_path)
    if relative.is_absolute() or ".." in relative.parts or "" in relative.parts:
        raise UnsafePathError(f"unsafe artifact path: {artifact_path!r}")
    parts = [part for part in relative.parts if part != "."]
    if not parts:
        raise UnsafePathError(f"artifact path must not be empty: {artifact_path!r}")
    destination = root.joinpath(*parts).resolve()
    root = root.resolve()
    if root not in destination.parents:
        raise UnsafePathError(f"artifact path escapes destination: {artifact_path!r}")
    return destination


def _tar_directory(source: Path, destination: Path, artifact_path: str) -> None:
    archive_root = PurePosixPath(artifact_path)
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        paths = [source, *sorted(source.rglob("*"), key=lambda item: item.as_posix())]
        for path in paths:
            if path.is_symlink():
                raise InvalidModelPackError(
                    f"directory layers must not contain symbolic links: {path}"
                )
            relative = archive_root / PurePosixPath(
                path.relative_to(source).as_posix()
            )
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if path.is_file():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def _uncompressed_digest(path: Path, media_type: str) -> str:
    if media_type.endswith("+gzip"):
        digest = hashlib.sha256()
        with gzip.open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    if media_type.endswith("+zstd"):
        import zstandard

        digest = hashlib.sha256()
        with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(
            source
        ) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    return _sha256_file(path)


def _apply_tar_metadata(path: Path, member: tarfile.TarInfo) -> None:
    try:
        path.chmod(member.mode & 0o777)
        os.utime(path, (member.mtime, member.mtime))
    except (OSError, TypeError, ValueError) as error:
        raise InvalidModelPackError(
            f"invalid metadata for tar member: {member.name!r}"
        ) from error


def _extract_tar_archive(
    archive: tarfile.TarFile,
    destination: Path,
    boundary: Path,
    overwrite: bool,
) -> list[Path]:
    files: list[Path] = []
    directories: list[tuple[Path, tarfile.TarInfo]] = []
    for member in archive:
        normalized_name = member.name.rstrip("/")
        if normalized_name in ("", ".") and member.isdir():
            continue
        if member.issym() or member.islnk():
            raise UnsafePathError(
                f"links are not allowed in ModelPack tar layers: {member.name!r}"
            )
        if not member.isfile() and not member.isdir():
            raise UnsafePathError(
                f"unsupported tar member type in ModelPack layer: {member.name!r}"
            )

        target = _safe_destination(destination, member.name)
        if boundary not in target.parents:
            raise UnsafePathError(f"tar member escapes output root: {member.name!r}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            directories.append((target, member))
            continue
        if not overwrite and target.exists():
            raise FileExistsError(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise InvalidModelPackError(
                f"cannot read tar member contents: {member.name!r}"
            )
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "wb") as output, source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            _apply_tar_metadata(temporary_path, member)
            if not overwrite and target.exists():
                raise FileExistsError(target)
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        files.append(target)

    for path, member in reversed(directories):
        _apply_tar_metadata(path, member)
    return files


def _extract_tar_safely(
    archive_path: Path,
    destination: Path,
    mode: str,
    boundary: Path,
    overwrite: bool = True,
) -> list[Path]:
    boundary = boundary.resolve()
    destination = destination.resolve()
    if boundary not in (destination, *destination.parents):
        raise UnsafePathError(f"tar destination escapes output root: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode) as archive:
        return _extract_tar_archive(archive, destination, boundary, overwrite)


def _extract_zstd_tar_safely(
    archive_path: Path,
    destination: Path,
    boundary: Path,
    overwrite: bool = True,
) -> list[Path]:
    import zstandard

    boundary = boundary.resolve()
    destination = destination.resolve()
    if boundary not in (destination, *destination.parents):
        raise UnsafePathError(f"tar destination escapes output root: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with archive_path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(
        source
    ) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        return _extract_tar_archive(archive, destination, boundary, overwrite)


class ModelPackClient:
    """Push and pull ModelPack artifacts from OCI-compatible registries."""

    def __init__(
        self,
        hostname: str | None = None,
        *,
        insecure: bool = False,
        tls_verify: bool | str = True,
        auth_backend: str = "token",
        oras_client: Any | None = None,
    ) -> None:
        self._client = oras_client or oras.client.OrasClient(
            hostname=hostname,
            insecure=insecure,
            tls_verify=tls_verify,
            auth_backend=auth_backend,
        )

    def login(
        self,
        hostname: str,
        username: str,
        password: str,
        *,
        config_path: str | Path | None = None,
    ) -> Mapping[str, Any]:
        return self._client.login(
            username=username,
            password=password,
            hostname=hostname,
            config_path=str(config_path) if config_path else None,
        )

    def logout(self, hostname: str) -> None:
        self._client.logout(hostname)

    def push(
        self,
        reference: str,
        layers: Iterable[ModelLayer | str | Path],
        config: ModelPackConfig | Mapping[str, Any] | str | Path,
        *,
        annotations: Mapping[str, str] | None = None,
        config_path: str | Path | None = None,
        chunked: bool | None = None,
        chunk_size: int = 16 * 1024 * 1024,
    ) -> PushResult:
        """Build and push a specification-compliant ModelPack artifact."""
        container = self._client.get_container(reference)
        self._client.auth.load_configs(
            container, configs=[str(config_path)] if config_path else None
        )
        manifest: dict[str, Any] = {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST_MEDIA_TYPE,
            "artifactType": MODEL_MANIFEST_ARTIFACT_TYPE,
            "config": {},
            "layers": [],
        }
        if annotations:
            manifest["annotations"] = dict(annotations)

        with tempfile.TemporaryDirectory(prefix="modelpack-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            prepared: list[tuple[Path, str, dict[str, str]]] = []
            for index, item in enumerate(layers):
                layer = item if isinstance(item, ModelLayer) else ModelLayer(item)
                source = Path(layer.path).expanduser().resolve()
                if not source.exists():
                    raise FileNotFoundError(source)
                media_type = layer.media_type
                if media_type not in MODEL_LAYER_MEDIA_TYPES:
                    raise InvalidModelPackError(
                        f"unsupported ModelPack layer media type: {media_type}"
                    )
                artifact_path = layer.artifact_path or source.name
                _safe_destination(Path("/modelpack"), artifact_path)
                layer_annotations = dict(layer.annotations)
                layer_annotations.setdefault(FILEPATH_ANNOTATION, artifact_path)
                layer_annotations.setdefault(OCI_TITLE_ANNOTATION, artifact_path)
                layer_annotations.setdefault(
                    FILE_METADATA_ANNOTATION, _file_metadata(source)
                )

                if source.is_dir():
                    if media_type == WEIGHT_RAW_MEDIA_TYPE:
                        media_type = WEIGHT_TAR_MEDIA_TYPE
                    if media_type not in TAR_MEDIA_TYPES or media_type.endswith(
                        ("+gzip", "+zstd")
                    ):
                        raise InvalidModelPackError(
                            "directories require an uncompressed ModelPack tar media type"
                        )
                    packed = temporary_root / f"layer-{index}.tar"
                    _tar_directory(source, packed, artifact_path)
                    source = packed
                prepared.append((source, media_type, layer_annotations))

            diff_ids = [
                _uncompressed_digest(path, media_type)
                for path, media_type, _ in prepared
            ]
            config_document = self._build_config(config, diff_ids)
            config_file = temporary_root / "config.json"
            config_file.write_text(
                json.dumps(config_document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

            for source, media_type, layer_annotations in prepared:
                descriptor = oras.oci.NewLayer(str(source), media_type=media_type)
                descriptor["annotations"] = layer_annotations
                response = self._client.upload_blob(
                    str(source),
                    container,
                    descriptor,
                    do_chunked=(
                        chunked
                        if chunked is not None
                        else descriptor["size"] > CHUNKED_UPLOAD_THRESHOLD
                    ),
                    chunk_size=chunk_size,
                )
                self._client._check_200_response(response)
                manifest["layers"].append(descriptor)

            config_descriptor, _ = oras.oci.ManifestConfig(
                str(config_file), MODEL_CONFIG_MEDIA_TYPE
            )
            response = self._client.upload_blob(
                str(config_file), container, config_descriptor
            )
            self._client._check_200_response(response)
            manifest["config"] = config_descriptor
            self._validate_manifest(manifest)

            response, manifest_digest = self._upload_manifest(container, manifest)
            self._client._check_200_response(response)
            reported = response.headers.get("Docker-Content-Digest")
            if reported and manifest_digest and reported != manifest_digest:
                raise InvalidModelPackError(
                    f"manifest digest mismatch: registry reported {reported}, "
                    f"expected {manifest_digest}"
                )
            return PushResult(
                reference=str(container),
                digest=manifest_digest or reported,
                manifest=deepcopy(manifest),
            )

    def pull(
        self,
        reference: str,
        destination: str | Path,
        *,
        config_path: str | Path | None = None,
        overwrite: bool = True,
        unpack: bool = True,
    ) -> PullResult:
        """Pull and validate a ModelPack artifact, including its model config."""
        container = self._client.get_container(reference)
        self._client.auth.load_configs(
            container, configs=[str(config_path)] if config_path else None
        )
        manifest = self._get_manifest(container, reference)
        self._validate_manifest(manifest)

        output_root = Path(destination).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        config_file = output_root / ".modelpack-config.json"
        self._download(container, manifest["config"]["digest"], config_file, overwrite)
        try:
            config_document = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InvalidModelPackError("ModelPack config is not valid JSON") from error
        self._validate_config(config_document, len(manifest["layers"]))
        diff_ids = config_document["modelfs"]["diffIds"]

        files: list[Path] = []
        for index, layer in enumerate(manifest["layers"]):
            annotations = layer.get("annotations") or {}
            artifact_path = annotations.get(FILEPATH_ANNOTATION) or annotations.get(
                OCI_TITLE_ANNOTATION
            )
            if not artifact_path:
                artifact_path = f"layer-{index}"
            destination_path = _safe_destination(output_root, artifact_path)
            media_type = layer["mediaType"]

            with tempfile.NamedTemporaryFile(
                prefix=".modelpack-layer-", dir=output_root, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                self._download(container, layer["digest"], temporary_path, overwrite=True)
                actual_diff_id = _uncompressed_digest(temporary_path, media_type)
                if actual_diff_id != diff_ids[index]:
                    raise InvalidModelPackError(
                        f"layer DiffID mismatch: expected {diff_ids[index]}, "
                        f"got {actual_diff_id}"
                    )

                if unpack and media_type in TAR_MEDIA_TYPES:
                    if media_type.endswith("+zstd"):
                        extracted = _extract_zstd_tar_safely(
                            temporary_path,
                            output_root,
                            boundary=output_root,
                            overwrite=overwrite,
                        )
                    else:
                        mode = "r:gz" if media_type.endswith("+gzip") else "r:"
                        extracted = _extract_tar_safely(
                            temporary_path,
                            output_root,
                            mode,
                            boundary=output_root,
                            overwrite=overwrite,
                        )
                    files.extend(extracted)
                    continue

                if destination_path.exists() and not overwrite:
                    raise FileExistsError(destination_path)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_path, destination_path)
                _apply_file_metadata(
                    destination_path, annotations.get(FILE_METADATA_ANNOTATION)
                )
                files.append(destination_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        return PullResult(
            reference=str(container),
            files=tuple(files),
            config=config_document,
            manifest=manifest,
        )

    def _manifest_url(self, container: Any) -> str | None:
        manifest_url = getattr(container, "manifest_url", None)
        prefix = getattr(self._client, "prefix", None)
        if not callable(manifest_url) or not prefix:
            return None
        return f"{prefix}://{manifest_url()}"

    def _get_manifest(self, container: Any, reference: str) -> dict[str, Any]:
        """Fetch the manifest, verifying it against a digest reference when given."""
        expected = _reference_digest(reference)
        url = self._manifest_url(container)
        do_request = getattr(self._client, "do_request", None)
        if url is None or not callable(do_request):
            if expected:
                raise InvalidModelPackError(
                    "cannot verify a digest reference without raw manifest access"
                )
            return self._client.get_manifest(
                container, allowed_media_type=[OCI_MANIFEST_MEDIA_TYPE]
            )

        response = do_request(
            url, "GET", headers={"Accept": OCI_MANIFEST_MEDIA_TYPE}
        )
        self._client._check_200_response(response)
        payload = response.content
        actual = "sha256:" + hashlib.sha256(payload).hexdigest()
        if expected and actual != expected:
            raise InvalidModelPackError(
                f"manifest digest mismatch: expected {expected}, got {actual}"
            )
        reported = response.headers.get("Docker-Content-Digest")
        if reported and reported != actual:
            raise InvalidModelPackError(
                f"manifest digest mismatch: registry reported {reported}, got {actual}"
            )
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as error:
            raise InvalidModelPackError("manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise InvalidModelPackError("manifest must be a JSON object")
        return manifest

    def _upload_manifest(
        self, container: Any, manifest: Mapping[str, Any]
    ) -> tuple[Any, str | None]:
        """Upload the manifest as exact bytes so its digest is unambiguous."""
        url = self._manifest_url(container)
        do_request = getattr(self._client, "do_request", None)
        if url is None or not callable(do_request):
            return self._client.upload_manifest(manifest, container), None
        payload = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        response = do_request(
            url,
            "PUT",
            headers={
                "Content-Type": OCI_MANIFEST_MEDIA_TYPE,
                "Content-Length": str(len(payload)),
            },
            data=payload,
        )
        return response, "sha256:" + hashlib.sha256(payload).hexdigest()

    def _download(
        self, container: Any, digest: str, destination: Path, overwrite: bool
    ) -> None:
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(temporary_fd)
        temporary_path = Path(temporary_name)
        try:
            self._client.download_blob(container, digest, str(temporary_path))
            actual_digest = _sha256_file(temporary_path)
            if actual_digest != digest:
                raise InvalidModelPackError(
                    f"blob digest mismatch: expected {digest}, got {actual_digest}"
                )
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _build_config(
        config: ModelPackConfig | Mapping[str, Any] | str | Path,
        diff_ids: list[str],
    ) -> dict[str, Any]:
        if isinstance(config, ModelPackConfig):
            document = config.to_dict(diff_ids)
        elif isinstance(config, (str, Path)):
            with Path(config).expanduser().open(encoding="utf-8") as stream:
                document = json.load(stream)
        else:
            document = deepcopy(dict(config))
        document["modelfs"] = {"type": "layers", "diffIds": diff_ids}
        ModelPackClient._validate_config(document, len(diff_ids))
        return document

    @staticmethod
    def _validate_config(config: Mapping[str, Any], layer_count: int) -> None:
        if not isinstance(config.get("descriptor"), Mapping):
            raise InvalidModelPackError("config.descriptor must be an object")
        if not isinstance(config.get("config"), Mapping):
            raise InvalidModelPackError("config.config must be an object")
        modelfs = config.get("modelfs")
        if not isinstance(modelfs, Mapping) or modelfs.get("type") != "layers":
            raise InvalidModelPackError("config.modelfs.type must be 'layers'")
        diff_ids = modelfs.get("diffIds")
        if not isinstance(diff_ids, list) or len(diff_ids) != layer_count:
            raise InvalidModelPackError(
                "config.modelfs.diffIds must match the manifest layer count"
            )
        if not all(
            isinstance(digest, str)
            and len(digest) == 71
            and digest.startswith("sha256:")
            and all(character in "0123456789abcdef" for character in digest[7:])
            for digest in diff_ids
        ):
            raise InvalidModelPackError("all diffIds must be sha256 digests")

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        if manifest.get("schemaVersion") != 2:
            raise InvalidModelPackError("manifest schemaVersion must be 2")
        if manifest.get("mediaType") != OCI_MANIFEST_MEDIA_TYPE:
            raise InvalidModelPackError("artifact is not an OCI image manifest")
        if manifest.get("artifactType") != MODEL_MANIFEST_ARTIFACT_TYPE:
            raise InvalidModelPackError("artifact is not a CNCF ModelPack")
        config = manifest.get("config")
        if (
            not isinstance(config, Mapping)
            or config.get("mediaType") != MODEL_CONFIG_MEDIA_TYPE
        ):
            raise InvalidModelPackError("manifest has no ModelPack config descriptor")
        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise InvalidModelPackError("manifest.layers must be an array")
        for layer in layers:
            if (
                not isinstance(layer, Mapping)
                or layer.get("mediaType") not in MODEL_LAYER_MEDIA_TYPES
            ):
                raise InvalidModelPackError("manifest contains a non-ModelPack layer")


__all__ = ["ModelPackClient"]
