"""CNCF ModelPack client backed by an OCI registry."""

from __future__ import annotations

import errno
import functools
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import stat
import tarfile
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import jsonschema
import oras.auth
import oras.client
import oras.container
import oras.utils
import zstandard

from ..constants import (
    CODE_RAW_MEDIA_TYPE,
    CODE_TAR_MEDIA_TYPE,
    DATASET_RAW_MEDIA_TYPE,
    DATASET_TAR_MEDIA_TYPE,
    DOC_RAW_MEDIA_TYPE,
    DOC_TAR_MEDIA_TYPE,
    FILE_METADATA_ANNOTATION,
    FILEPATH_ANNOTATION,
    MODEL_CONFIG_MEDIA_TYPE,
    MODEL_LAYER_MEDIA_TYPES,
    MODEL_MANIFEST_ARTIFACT_TYPE,
    OCI_MANIFEST_MEDIA_TYPE,
    OCI_TITLE_ANNOTATION,
    TAR_MEDIA_TYPES,
    WEIGHT_CONFIG_RAW_MEDIA_TYPE,
    WEIGHT_CONFIG_TAR_MEDIA_TYPE,
    WEIGHT_RAW_MEDIA_TYPE,
    WEIGHT_TAR_MEDIA_TYPE,
)
from ..errors import InvalidModelPackError, ModelPackError, UnsafePathError
from ..models import ModelLayer, ModelPackConfig, PullResult, PushResult
from ..schema import CONFIG_SCHEMA

_CHUNK_SIZE = 1024 * 1024

_COMPRESSED_SUFFIXES = ("+gzip", "+zstd")

#: The tar media type used when a layer with a raw media type is a directory.
_DIRECTORY_MEDIA_TYPES = {
    WEIGHT_RAW_MEDIA_TYPE: WEIGHT_TAR_MEDIA_TYPE,
    WEIGHT_CONFIG_RAW_MEDIA_TYPE: WEIGHT_CONFIG_TAR_MEDIA_TYPE,
    DOC_RAW_MEDIA_TYPE: DOC_TAR_MEDIA_TYPE,
    CODE_RAW_MEDIA_TYPE: CODE_TAR_MEDIA_TYPE,
    DATASET_RAW_MEDIA_TYPE: DATASET_TAR_MEDIA_TYPE,
}

#: File name extensions for tar layers saved without unpacking; the first is added.
_ARCHIVE_EXTENSIONS = {
    "+gzip": (".tar.gz", ".tgz"),
    "+zstd": (".tar.zst", ".tzst"),
    "": (".tar",),
}

# modctl writes the numbers 0 and 5, not the tar header characters "0" and "5".
_TYPEFLAG_REGULAR = 0
_TYPEFLAG_DIRECTORY = 5

_DIGEST_PATTERN = re.compile(r"sha256:[a-f0-9]{64}|sha512:[a-f0-9]{128}")
_RFC3339_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2})[Tt](\d{2}:\d{2}:\d{2})(?:\.(\d+))?([Zz]|[+-]\d{2}:\d{2})"
)

_ARCHIVE_ERRORS = (
    tarfile.TarError,
    gzip.BadGzipFile,
    zlib.error,
    zstandard.ZstdError,
    EOFError,
)

#: The OCI distribution specification asks clients to accept manifests up to 4 MiB.
_MAX_MANIFEST_SIZE = 4 * 1024 * 1024

_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_PATTERN.fullmatch(value) is not None


def _algorithm(digest: str) -> str:
    return digest.partition(":")[0]


def _digest_bytes(payload: bytes, algorithm: str = "sha256") -> str:
    return f"{algorithm}:{hashlib.new(algorithm, payload).hexdigest()}"


def _is_empty_blob(digest: str) -> bool:
    return digest == _digest_bytes(b"", _algorithm(digest))


def _digest_file(path: Path, algorithm: str = "sha256") -> tuple[str, int]:
    """Return the digest and size of a file, read in a single pass."""
    with path.open("rb") as stream:
        reader = _HashingReader(stream, algorithm)
        return reader.digest_to_end(), reader.size


def _new_file(directory: Path, prefix: str) -> Path:
    """Create an empty file in ``directory`` named ``prefix`` plus random characters.

    ``tempfile`` creates files only their owner can read. This one gets the
    permissions the umask gives any new file, which a pulled file keeps unless its
    layer carries a mode.
    """
    while True:
        path = directory / f"{prefix}{secrets.token_hex(8)}"
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            continue
        return path


def _reference_digest(reference: str) -> str | None:
    """Return the digest of a ``repo@<algorithm>:<hex>`` reference, if present."""
    repository, separator, digest = reference.rpartition("@")
    if not separator or not repository:
        return None
    if not _is_digest(digest):
        raise InvalidModelPackError(f"unsupported digest reference: {digest!r}")
    return digest


def _parse_rfc3339(value: str) -> datetime:
    """Parse an RFC 3339 timestamp, such as the RFC3339Nano ones Go tools write.

    ``datetime.fromisoformat()`` on Python 3.10 accepts neither ``Z`` nor fractions
    other than 3 or 6 digits, so the fraction is cut to microseconds first.
    """
    match = _RFC3339_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid RFC 3339 timestamp: {value!r}")
    date, time, fraction, offset = match.groups()
    microseconds = f".{fraction[:6].ljust(6, '0')}" if fraction else ""
    if offset in ("Z", "z"):
        offset = "+00:00"
    return datetime.fromisoformat(f"{date}T{time}{microseconds}{offset}")


# jsonschema only checks "date-time" when an optional extra is installed.
_FORMAT_CHECKER = jsonschema.FormatChecker(formats=())


@_FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _is_date_time(value: object) -> bool:
    return not isinstance(value, str) or _parse_rfc3339(value) is not None


_CONFIG_VALIDATOR = jsonschema.Draft202012Validator(
    CONFIG_SCHEMA, format_checker=_FORMAT_CHECKER
)


def _file_metadata(path: Path, name: str) -> str:
    info = path.stat()
    return json.dumps(
        {
            "name": name,
            "mode": stat.S_IMODE(info.st_mode) & 0o777,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "size": info.st_size,
            "mtime": datetime.fromtimestamp(info.st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "typeflag": (
                _TYPEFLAG_DIRECTORY if stat.S_ISDIR(info.st_mode) else _TYPEFLAG_REGULAR
            ),
        },
        separators=(",", ":"),
    )


def _safe_file_mode(mode: int) -> int:
    """Return the permission bits to give a pulled file with a registry-supplied mode.

    As with tarfile's "data" filter, the owner can always read and write the file,
    and setuid/setgid/sticky and group/other write bits are dropped.
    """
    return (mode & 0o755) | 0o600


def _apply_file_metadata(path: Path, annotation: str | None) -> None:
    if not annotation:
        return
    try:
        metadata = json.loads(annotation)
        if not isinstance(metadata, dict):
            raise TypeError("file metadata must be a JSON object")
        if metadata.get("typeflag") == _TYPEFLAG_DIRECTORY:
            # Describes the directory a tar layer unpacks to, not the saved archive.
            return
        mode = metadata.get("mode")
        mtime = metadata.get("mtime")
        if isinstance(mode, int) and not isinstance(mode, bool):
            path.chmod(_safe_file_mode(mode))
        if isinstance(mtime, str):
            timestamp = _parse_rfc3339(mtime).timestamp()
            os.utime(path, (timestamp, timestamp))
    except (OSError, OverflowError, TypeError, ValueError) as error:
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


def _archive_path(artifact_path: str, media_type: str) -> str:
    """Return where a tar layer is saved when it is not unpacked."""
    compression = next(
        (suffix for suffix in _COMPRESSED_SUFFIXES if media_type.endswith(suffix)), ""
    )
    extensions = _ARCHIVE_EXTENSIONS[compression]
    if artifact_path.endswith(extensions):
        return artifact_path
    return artifact_path.rstrip("/") + extensions[0]


def _check_replaceable(path: Path, overwrite: bool) -> None:
    if path.is_dir():
        raise IsADirectoryError(
            errno.EISDIR, "a layer file cannot replace a directory", str(path)
        )
    if not overwrite and path.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))


def _check_diff_id(expected: str, actual: str) -> None:
    if actual != expected:
        raise InvalidModelPackError(
            f"layer DiffID mismatch: expected {expected}, got {actual}"
        )


def _check_reported_digest(response: Any, payload: bytes) -> None:
    """Compare the Docker-Content-Digest a registry reported with the manifest bytes."""
    reported = response.headers.get("Docker-Content-Digest")
    if _is_digest(reported):
        actual = _digest_bytes(payload, _algorithm(reported))
        if actual != reported:
            raise InvalidModelPackError(
                f"manifest digest mismatch: registry reported {reported}, got {actual}"
            )


class _HashingWriter:
    """Write-only stream that hashes and counts what it passes on to ``stream``."""

    def __init__(self, stream: Any, algorithm: str = "sha256") -> None:
        self._stream = stream
        self._algorithm = algorithm
        self._hasher = hashlib.new(algorithm)
        self.size = 0

    def write(self, data: bytes) -> int:
        self._stream.write(data)
        self._hasher.update(data)
        self.size += len(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    @property
    def digest(self) -> str:
        return f"{self._algorithm}:{self._hasher.hexdigest()}"


class _HashingReader:
    """Read-only stream that hashes and counts what is read from ``stream``."""

    def __init__(self, stream: Any, algorithm: str) -> None:
        self._stream = stream
        self._algorithm = algorithm
        self._hasher = hashlib.new(algorithm)
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._hasher.update(data)
        self.size += len(data)
        return data

    def digest_to_end(self) -> str:
        """Read the rest of the stream, such as tar padding, and return its digest."""
        while self.read(_CHUNK_SIZE):
            pass
        return f"{self._algorithm}:{self._hasher.hexdigest()}"


class _BlobReader(_HashingReader):
    """Reads a blob, never more than one byte past the size its descriptor gives."""

    def __init__(self, stream: Any, descriptor: Mapping[str, Any]) -> None:
        super().__init__(stream, _algorithm(descriptor["digest"]))
        self._expected_digest = descriptor["digest"]
        self._expected_size = descriptor["size"]

    def read(self, size: int = -1) -> bytes:
        # One byte past the end is enough to tell that a blob is too long.
        remaining = self._expected_size - self.size + 1
        data = super().read(remaining if size < 0 else min(size, remaining))
        if self.size > self._expected_size:
            raise InvalidModelPackError(
                f"blob {self._expected_digest} is longer than its descriptor size of "
                f"{self._expected_size} bytes"
            )
        return data

    def verify(self) -> None:
        """Read the rest of the blob and check its size and digest."""
        actual = self.digest_to_end()
        if self.size != self._expected_size:
            raise InvalidModelPackError(
                f"blob size mismatch: expected {self._expected_size} bytes, "
                f"got {self.size}"
            )
        if actual != self._expected_digest:
            raise InvalidModelPackError(
                f"blob digest mismatch: expected {self._expected_digest}, got {actual}"
            )


class _ResponseStream:
    """Read-only stream over a streamed response body.

    A read returns at most one chunk as it arrived, so it can be shorter than asked.
    """

    def __init__(self, response: Any) -> None:
        self._chunks = response.iter_content(chunk_size=_CHUNK_SIZE)
        self._pending = memoryview(b"")

    def read(self, size: int = -1) -> bytes:
        if not self._pending:
            self._pending = memoryview(next(self._chunks, b""))
        if size < 0:
            size = len(self._pending)
        data, self._pending = self._pending[:size], self._pending[size:]
        return bytes(data)


class _ReplayableBody:
    """File-backed request body that can be sent more than once.

    oras's ``do_request`` resends the same ``data`` after answering an auth
    challenge. requests sizes a body with ``len()`` every time it prepares a request,
    so rewinding there makes a retried PUT send the whole file again. Being iterable
    with ``tell()`` makes requests record the start of the body, which it seeks back
    to when it follows a 307 or 308 redirect.
    """

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self._stream = stream
        self._size = size

    def __len__(self) -> int:
        self._stream.seek(0)
        return self._size

    def __iter__(self) -> Iterator[bytes]:
        return iter(lambda: self._stream.read(_CHUNK_SIZE), b"")

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._stream.seek(offset, whence)


@contextmanager
def _compressing(stream: Any, media_type: str) -> Iterator[Any]:
    if media_type.endswith("+gzip"):
        # An empty name and a zero mtime keep the gzip header reproducible.
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as compressed:
            yield compressed
    elif media_type.endswith("+zstd"):
        with zstandard.ZstdCompressor().stream_writer(
            stream, closefd=False
        ) as compressed:
            yield compressed
    else:
        yield stream


@contextmanager
def _decompressing(stream: BinaryIO, media_type: str) -> Iterator[Any]:
    if media_type.endswith("+gzip"):
        with gzip.GzipFile(mode="rb", fileobj=stream) as decompressed:
            yield decompressed
    elif media_type.endswith("+zstd"):
        with zstandard.ZstdDecompressor().stream_reader(
            stream, read_across_frames=True, closefd=False
        ) as decompressed:
            yield decompressed
    else:
        yield stream


def _uncompressed_digest(path: Path, media_type: str, algorithm: str = "sha256") -> str:
    try:
        with path.open("rb") as source, _decompressing(source, media_type) as stream:
            return _HashingReader(stream, algorithm).digest_to_end()
    except _ARCHIVE_ERRORS as error:
        raise InvalidModelPackError(f"cannot decompress layer: {path}") from error


def _digest_layer_file(path: Path, media_type: str) -> tuple[str, int, str]:
    """Return the digest, size and DiffID of a file layer, reading the file once."""
    try:
        with path.open("rb") as source:
            blob = _HashingReader(source, "sha256")
            diff_id = None
            if media_type.endswith(_COMPRESSED_SUFFIXES):
                with _decompressing(blob, media_type) as stream:
                    diff_id = _HashingReader(stream, "sha256").digest_to_end()
            digest = blob.digest_to_end()
    except _ARCHIVE_ERRORS as error:
        raise InvalidModelPackError(f"cannot decompress layer: {path}") from error
    return digest, blob.size, diff_id or digest


def _add_tar_member(archive: tarfile.TarFile, path: Path, name: str) -> None:
    info = path.lstat()
    member = tarfile.TarInfo(name)
    member.mtime = 0
    if stat.S_ISDIR(info.st_mode):
        member.type = tarfile.DIRTYPE
        member.mode = 0o755
        archive.addfile(member)
    elif stat.S_ISREG(info.st_mode):
        # Every hard link is stored as a regular file, because pull rejects links.
        member.type = tarfile.REGTYPE
        member.mode = 0o755 if info.st_mode & 0o111 else 0o644
        member.size = info.st_size
        with path.open("rb") as stream:
            archive.addfile(member, stream)
    elif stat.S_ISLNK(info.st_mode):
        raise InvalidModelPackError(
            f"directory layers must not contain symbolic links: {path}"
        )
    else:
        raise InvalidModelPackError(
            f"directory layers may only contain regular files and directories: {path}"
        )


def _pack_directory(
    source: Path, destination: Path, artifact_path: str, media_type: str
) -> tuple[str, str, int]:
    """Pack a directory as a reproducible tar layer.

    Entries are sorted and their owners, mtimes and modes normalized, so the same
    tree gets the same digest on every machine. Returns the blob digest, the DiffID
    (the digest of the uncompressed tar) and the blob size.
    """
    archive_root = PurePosixPath(artifact_path)
    paths = [source, *sorted(source.rglob("*"), key=lambda item: item.as_posix())]
    with destination.open("wb") as output:
        blob = _HashingWriter(output)
        with _compressing(blob, media_type) as compressed:
            tar_stream = blob if compressed is blob else _HashingWriter(compressed)
            with tarfile.open(
                fileobj=tar_stream, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                for path in paths:
                    relative = PurePosixPath(path.relative_to(source).as_posix())
                    _add_tar_member(archive, path, (archive_root / relative).as_posix())
    return blob.digest, tar_stream.digest, blob.size


def _apply_tar_metadata(path: Path, member: tarfile.TarInfo) -> None:
    try:
        path.chmod(_safe_file_mode(member.mode))
        os.utime(path, (member.mtime, member.mtime))
    except (OSError, OverflowError, TypeError, ValueError) as error:
        raise InvalidModelPackError(
            f"invalid metadata for tar member: {member.name!r}"
        ) from error


def _stage_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    overwrite: bool,
) -> tuple[Path, Path] | None:
    """Write a file member next to its target; return (temporary, target)."""
    if member.name.rstrip("/") in ("", ".") and member.isdir():
        return None
    if member.issym() or member.islnk():
        raise UnsafePathError(
            f"links are not allowed in ModelPack tar layers: {member.name!r}"
        )
    if not member.isfile() and not member.isdir():
        raise UnsafePathError(
            f"unsupported tar member type in ModelPack layer: {member.name!r}"
        )

    target = _safe_destination(destination, member.name)
    if member.isdir():
        # Directory modes and mtimes are not applied: a read-only directory would
        # block later members and layers, and the directory may be the user's own.
        target.mkdir(parents=True, exist_ok=True)
        return None
    _check_replaceable(target, overwrite)

    target.parent.mkdir(parents=True, exist_ok=True)
    contents = archive.extractfile(member)
    if contents is None:
        raise InvalidModelPackError(f"cannot read tar member contents: {member.name!r}")
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as output, contents:
            shutil.copyfileobj(contents, output, length=_CHUNK_SIZE)
        _apply_tar_metadata(temporary_path, member)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, target


def _extract_tar_layer(
    blob: _BlobReader,
    media_type: str,
    destination: Path,
    overwrite: bool,
    diff_id: str | None,
) -> list[Path]:
    """Unpack a tar layer read from ``blob`` into ``destination``.

    Files are moved into place only after the whole blob has been read and matched
    its digest and, when ``diff_id`` is given, the uncompressed tar has matched it.
    That way a layer is read once, straight from the registry, rather than saved,
    verified and then unpacked.
    """
    staged: list[tuple[Path, Path]] = []
    targets: set[Path] = set()
    try:
        with _decompressing(blob, media_type) as stream:
            reader = _HashingReader(stream, _algorithm(diff_id)) if diff_id else stream
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    staged_member = _stage_tar_member(
                        archive, member, destination, overwrite
                    )
                    if staged_member is None:
                        continue
                    staged.append(staged_member)
                    if staged_member[1] in targets:
                        raise InvalidModelPackError(
                            f"tar layer contains {member.name!r} more than once"
                        )
                    targets.add(staged_member[1])
            actual_diff_id = reader.digest_to_end() if diff_id else None
        blob.verify()
        if diff_id:
            _check_diff_id(diff_id, actual_diff_id)

        # Check every target before moving any, so a conflict leaves nothing placed.
        for _, target in staged:
            _check_replaceable(target, overwrite)
        for temporary_path, target in staged:
            os.replace(temporary_path, target)
        return [target for _, target in staged]
    except _ARCHIVE_ERRORS as error:
        raise InvalidModelPackError(f"invalid tar layer: {error}") from error
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _PreparedLayer:
    path: Path
    descriptor: dict[str, Any]
    diff_id: str


def _layer_name(layer: ModelLayer) -> str:
    # Name the layer after the path as given: a symlink, such as a file in a
    # Hugging Face cache snapshot, must not be named after the blob it points to.
    return Path(os.path.abspath(Path(layer.path).expanduser())).name


def _layer_file_path(layer: ModelLayer) -> str:
    """Return the path a layer is pulled to."""
    return layer.annotations.get(
        FILEPATH_ANNOTATION, layer.artifact_path or _layer_name(layer)
    )


def _check_unique_file_paths(layers: Iterable[ModelLayer]) -> None:
    counts = Counter(PurePosixPath(_layer_file_path(layer)) for layer in layers)
    duplicates = sorted(str(path) for path, count in counts.items() if count > 1)
    if duplicates:
        raise InvalidModelPackError(
            "more than one layer would be pulled to the same path: "
            + ", ".join(duplicates)
            + "; set artifact_path to tell them apart"
        )


def _prepare_layer(layer: ModelLayer, packed_path: Path) -> _PreparedLayer:
    given = Path(layer.path).expanduser()
    source = given.resolve()
    if not source.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(given))
    media_type = layer.media_type
    if media_type not in MODEL_LAYER_MEDIA_TYPES:
        raise InvalidModelPackError(f"unsupported ModelPack layer media type: {media_type}")

    name = _layer_name(layer)
    artifact_path = layer.artifact_path or name
    _safe_destination(Path("/modelpack"), artifact_path)
    annotations = dict(layer.annotations)
    annotations.setdefault(FILEPATH_ANNOTATION, artifact_path)
    annotations.setdefault(OCI_TITLE_ANNOTATION, artifact_path)
    annotations.setdefault(FILE_METADATA_ANNOTATION, _file_metadata(source, name))

    if source.is_dir():
        media_type = _DIRECTORY_MEDIA_TYPES.get(media_type, media_type)
        digest, diff_id, size = _pack_directory(
            source, packed_path, artifact_path, media_type
        )
        path = packed_path
    elif source.is_file():
        path = source
        digest, size, diff_id = _digest_layer_file(source, media_type)
    else:
        raise InvalidModelPackError(f"layers must be regular files or directories: {given}")

    return _PreparedLayer(
        path=path,
        descriptor={
            "mediaType": media_type,
            "digest": digest,
            "size": size,
            "annotations": annotations,
        },
        diff_id=diff_id,
    )


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme)


def _registry_origin(client: Any, container: Any) -> tuple[str, str, int | None]:
    return _origin(f"{client.prefix}://{container.registry}")


class _RegistryAuth:
    """Mixin that keeps an oras auth backend to the registry it was created for.

    oras answers any auth challenge with the backend's credentials, even one from
    another host that a request was redirected to. oras's token backend also answers
    a challenge by resending the token it already has, although a challenge means
    that token was rejected, for example because it was issued for pulling another
    repository, so resending it only fails again.
    """

    registry_origin: tuple[str, str, int | None]
    renew_tokens: bool

    def authenticate_request(
        self, original: Any, headers: dict, refresh: bool = False
    ) -> tuple[dict, bool]:
        if _origin(original.url) != self.registry_origin:
            # oras gives up on this exception at once, but retries others for minutes.
            raise oras.auth.AuthenticationException(
                f"{original.url} asked for credentials, which are only sent to the "
                "registry itself"
            )
        return super().authenticate_request(  # type: ignore[misc]
            original, headers, refresh=refresh or self.renew_tokens
        )


@functools.cache
def _registry_auth_class(backend: type) -> type:
    return type(f"Registry{backend.__name__}", (_RegistryAuth, backend), {})


def _load_credentials(auth: Any, container: Any, config_path: str | Path | None) -> None:
    """Load the credentials for ``container`` from Docker's file and ``config_path``.

    Entries in ``config_path`` take precedence. oras would merge the files in no
    particular order, so its backend is handed them already merged.
    """
    config: dict[str, Any] = {"auths": {}, "credHelpers": {}, "credsStore": None}
    for path in (oras.utils.find_docker_config(), config_path):
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        config["auths"].update(document.get("auths") or {})
        config["credHelpers"].update(document.get("credHelpers") or {})
        config["credsStore"] = document.get("credsStore") or config["credsStore"]
    auth._auth_config = config
    auth.load_configs(container)


def _forget_saved_credentials(hostname: str, config_path: str | Path | None) -> None:
    """Remove the credentials for ``hostname`` from a Docker credential file."""
    path = config_path or oras.utils.find_docker_config()
    if not path:
        return
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    auths = document.get("auths") or {}
    hosts = [host for host in oras.utils.iter_localhosts(hostname) if host in auths]
    if not hosts:
        return
    for host in hosts:
        del auths[host]
    # Replace the file in one step, so a failed write cannot leave it truncated.
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent="\t")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _request(
    client: Any,
    container: Any,
    url: str,
    method: str,
    *,
    headers: Mapping[str, str] | None = None,
    data: Any = None,
    stream: bool = False,
) -> Any:
    """Send a request with the oras client's headers.

    oras adds its token or Basic credentials to any URL it is given, so only requests
    to the registry itself go through it. Others, such as an upload location on a
    storage service, are sent without credentials.
    """
    request_headers = {**client.headers, **(headers or {})}
    if _origin(url) == _registry_origin(client, container):
        return client.do_request(
            url, method, data=data, headers=request_headers, stream=stream
        )
    request_headers = {
        name: value
        for name, value in request_headers.items()
        if name.lower() != "authorization"
    }
    return client.session.request(
        method, url, data=data, headers=request_headers, stream=stream
    )


def _check_response(response: Any) -> None:
    """Raise ValueError, as oras does, for a response without a 2xx status."""
    if 200 <= response.status_code < 300:
        return
    message = f"{response.url} returned {response.status_code} {response.reason}"
    try:
        errors = response.json()["errors"]
        message += "".join(f"; {error['code']}: {error.get('message')}" for error in errors)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    raise ValueError(message)


def _upload_location(response: Any) -> str:
    location = response.headers.get("Location")
    if not location:
        raise ModelPackError("registry did not return a blob upload location")
    # A relative location refers to the URL of the request it answers.
    return urljoin(response.url, location)


def _start_upload(client: Any, container: Any) -> str:
    """Start a blob upload and return the URL its content goes to."""
    response = _request(
        client,
        container,
        f"{client.prefix}://{container.upload_blob_url()}",
        "POST",
        headers={"Content-Type": "application/octet-stream", "Content-Length": "0"},
    )
    _check_response(response)
    return _upload_location(response)


def _put_blob(
    client: Any, container: Any, path: Path, descriptor: Mapping[str, Any]
) -> Any:
    """Upload a blob as one streamed PUT, without reading it into memory."""
    upload_url = oras.utils.append_url_params(
        _start_upload(client, container), {"digest": descriptor["digest"]}
    )
    with path.open("rb") as stream:
        return _request(
            client,
            container,
            upload_url,
            "PUT",
            data=_ReplayableBody(stream, descriptor["size"]),
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(descriptor["size"]),
            },
        )


def _patch_blob(
    client: Any,
    container: Any,
    path: Path,
    descriptor: Mapping[str, Any],
    chunk_size: int,
) -> Any:
    """Upload a blob in PATCH requests of ``chunk_size`` bytes and a closing PUT."""
    upload_url = _start_upload(client, container)
    offset = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            response = _request(
                client,
                container,
                upload_url,
                "PATCH",
                data=chunk,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"{offset}-{offset + len(chunk) - 1}",
                },
            )
            _check_response(response)
            upload_url = _upload_location(response)
            offset += len(chunk)
    return _request(
        client,
        container,
        oras.utils.append_url_params(upload_url, {"digest": descriptor["digest"]}),
        "PUT",
        headers={"Content-Length": "0"},
    )


@contextmanager
def _open_blob(
    client: Any, container: Any, descriptor: Mapping[str, Any]
) -> Iterator[_BlobReader]:
    """Stream a blob. Call ``verify()`` on the reader once it has been consumed."""
    digest = descriptor["digest"]
    if _is_empty_blob(digest):
        # Some registries neither accept nor serve the empty blob, whose content is
        # known anyway.
        yield _BlobReader(io.BytesIO(), descriptor)
        return
    url = f"{client.prefix}://{container.get_blob_url(digest)}"
    with _request(client, container, url, "GET", stream=True) as response:
        _check_response(response)
        yield _BlobReader(_ResponseStream(response), descriptor)


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
        if oras_client is None and auth_backend not in oras.auth.auth_backends:
            raise ValueError(f"unknown oras auth backend: {auth_backend!r}")
        self._hostname = hostname
        self._insecure = insecure
        self._tls_verify = tls_verify
        self._auth_backend = auth_backend
        self._oras_client = oras_client
        self._credentials: dict[str, tuple[str, str]] = {}

    def login(
        self,
        hostname: str,
        username: str,
        password: str,
        *,
        config_path: str | Path | None = None,
    ) -> Mapping[str, Any]:
        if self._oras_client is not None:
            return self._oras_client.login(
                username=username,
                password=password,
                hostname=hostname,
                config_path=str(config_path) if config_path else None,
            )
        result = self._new_oras_client(hostname).login(
            username=username,
            password=password,
            hostname=hostname,
            config_path=str(config_path) if config_path else None,
        )
        self._credentials[hostname] = (username, password)
        return result

    def logout(
        self, hostname: str, *, config_path: str | Path | None = None
    ) -> None:
        """Forget the credentials for ``hostname``.

        Like ``docker logout``, this also removes them from the credential file that
        ``login()`` saved them to, so pass the same ``config_path``.
        """
        _forget_saved_credentials(hostname, config_path)
        if self._oras_client is not None:
            self._oras_client.logout(hostname)
            return
        self._credentials.pop(hostname, None)

    def push(
        self,
        reference: str,
        layers: Iterable[ModelLayer | str | Path],
        config: ModelPackConfig | Mapping[str, Any] | str | Path,
        *,
        annotations: Mapping[str, str] | None = None,
        config_path: str | Path | None = None,
        chunked: bool = False,
        chunk_size: int = 16 * 1024 * 1024,
    ) -> PushResult:
        """Build and push a specification-compliant ModelPack artifact.

        Each blob is streamed in a single PUT request. ``chunked=True`` uploads layers
        in ``chunk_size`` PATCH requests instead, which not every registry supports.
        """
        model_layers = [
            item if isinstance(item, ModelLayer) else ModelLayer(item) for item in layers
        ]
        _check_unique_file_paths(model_layers)
        client, container = self._registry_client(reference, config_path)
        with tempfile.TemporaryDirectory(prefix="modelpack-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            prepared = [
                _prepare_layer(layer, temporary_root / f"layer-{index}.tar")
                for index, layer in enumerate(model_layers)
            ]
            config_document = self._build_config(
                config, [layer.diff_id for layer in prepared]
            )
            config_bytes = json.dumps(
                config_document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            config_file = temporary_root / "config.json"
            config_file.write_bytes(config_bytes)

            manifest: dict[str, Any] = {
                "schemaVersion": 2,
                "mediaType": OCI_MANIFEST_MEDIA_TYPE,
                "artifactType": MODEL_MANIFEST_ARTIFACT_TYPE,
                "config": {
                    "mediaType": MODEL_CONFIG_MEDIA_TYPE,
                    "digest": _digest_bytes(config_bytes),
                    "size": len(config_bytes),
                },
                "layers": [layer.descriptor for layer in prepared],
            }
            if annotations:
                manifest["annotations"] = dict(annotations)
            self._validate_manifest(manifest)

            for layer in prepared:
                self._upload_blob(
                    client,
                    container,
                    layer.path,
                    layer.descriptor,
                    chunked=chunked,
                    chunk_size=chunk_size,
                )
            self._upload_blob(
                client,
                container,
                config_file,
                manifest["config"],
                chunked=False,
                chunk_size=chunk_size,
            )

        # The manifest is uploaded as exact bytes so its digest is unambiguous.
        payload = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        response = _request(
            client,
            container,
            f"{client.prefix}://{container.manifest_url()}",
            "PUT",
            headers={"Content-Type": OCI_MANIFEST_MEDIA_TYPE},
            data=payload,
        )
        _check_response(response)
        _check_reported_digest(response, payload)
        return PushResult(
            reference=str(container),
            digest=_digest_bytes(payload),
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
        """Pull and validate a ModelPack artifact, including its model config.

        Tar layers are unpacked into ``destination``. With ``unpack=False`` they are
        saved as archives named after their file path plus the archive extension,
        such as ``weights.tar``.
        """
        client, container = self._registry_client(reference, config_path)
        manifest = self._get_manifest(client, container, reference)
        self._validate_manifest(manifest)
        layers = manifest["layers"]

        output_root = Path(destination).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="modelpack-") as temporary_dir:
            config_file = Path(temporary_dir) / "config.json"
            self._download(client, container, manifest["config"], config_file)
            try:
                config_document = json.loads(config_file.read_bytes())
            except ValueError as error:
                raise InvalidModelPackError("ModelPack config is not valid JSON") from error
        self._validate_config(config_document, len(layers))
        diff_ids = config_document["modelfs"]["diffIds"]

        files: list[Path] = []
        for index, (layer, diff_id) in enumerate(zip(layers, diff_ids)):
            files.extend(
                self._pull_layer(
                    client,
                    container,
                    layer,
                    diff_id,
                    index,
                    output_root,
                    overwrite=overwrite,
                    unpack=unpack,
                )
            )
        return PullResult(
            reference=str(container),
            files=tuple(files),
            config=config_document,
            manifest=manifest,
        )

    def _pull_layer(
        self,
        client: Any,
        container: Any,
        layer: Mapping[str, Any],
        diff_id: str,
        index: int,
        output_root: Path,
        *,
        overwrite: bool,
        unpack: bool,
    ) -> list[Path]:
        annotations = layer.get("annotations") or {}
        artifact_path = (
            annotations.get(FILEPATH_ANNOTATION)
            or annotations.get(OCI_TITLE_ANNOTATION)
            or f"layer-{index}"
        )
        media_type = layer["mediaType"]
        is_tar = media_type in TAR_MEDIA_TYPES
        if is_tar and not unpack:
            artifact_path = _archive_path(artifact_path, media_type)
        destination_path = _safe_destination(output_root, artifact_path)
        if not (is_tar and unpack):
            _check_replaceable(destination_path, overwrite)

        blob_digest = layer["digest"]
        compressed = media_type.endswith(_COMPRESSED_SUFFIXES)
        # The DiffID of an uncompressed layer is its blob digest, which is verified
        # while downloading, so it only needs hashing again for another algorithm.
        diff_id_verified = not compressed and _algorithm(diff_id) == _algorithm(
            blob_digest
        )
        if diff_id_verified:
            _check_diff_id(diff_id, blob_digest)

        if is_tar and unpack:
            with _open_blob(client, container, layer) as blob:
                return _extract_tar_layer(
                    blob,
                    media_type,
                    output_root,
                    overwrite,
                    None if diff_id_verified else diff_id,
                )

        temporary_path = _new_file(output_root, ".modelpack-layer-")
        try:
            self._download(client, container, layer, temporary_path)
            if compressed:
                _check_diff_id(
                    diff_id,
                    _uncompressed_digest(temporary_path, media_type, _algorithm(diff_id)),
                )
            elif not diff_id_verified:
                _check_diff_id(
                    diff_id, _digest_file(temporary_path, _algorithm(diff_id))[0]
                )

            _apply_file_metadata(temporary_path, annotations.get(FILE_METADATA_ANNOTATION))
            _check_replaceable(destination_path, overwrite)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_path, destination_path)
            return [destination_path]
        finally:
            temporary_path.unlink(missing_ok=True)

    def _registry_client(
        self, reference: str, config_path: str | Path | None
    ) -> tuple[Any, Any]:
        """Return the oras client to use for ``reference`` and the parsed container."""
        if self._oras_client is not None:
            container = self._oras_client.get_container(reference)
            configs = [str(config_path)] if config_path else None
            self._oras_client.auth.load_configs(container, configs=configs)
            return self._oras_client, container

        container = oras.container.Container(reference, registry=self._hostname)
        # Every call gets a new oras client, so credential files are read again and a
        # token or Basic credentials never outlive a logout or reach another registry.
        client = self._new_oras_client(container.registry)
        _load_credentials(client.auth, container, config_path)
        credentials = self._credentials.get(container.registry)
        if credentials is not None:
            client.auth.set_basic_auth(*credentials)
        return client, container

    def _new_oras_client(self, registry: str) -> Any:
        client = oras.client.OrasClient(
            hostname=registry,
            insecure=self._insecure,
            tls_verify=self._tls_verify,
            auth_backend=self._auth_backend,
        )
        # Requests that skip oras, such as uploads to another host, use the session's.
        client.session.verify = self._tls_verify
        backend = type(client.auth)
        auth_class = _registry_auth_class(backend)
        auth = auth_class.__new__(auth_class)
        vars(auth).update(vars(client.auth))
        auth.registry_origin = _origin(f"{client.prefix}://{registry}")
        # Subclasses such as the ECR backend manage their own tokens.
        auth.renew_tokens = backend is oras.auth.TokenAuth
        client.auth = auth
        return client

    @staticmethod
    def _upload_blob(
        client: Any,
        container: Any,
        path: Path,
        descriptor: Mapping[str, Any],
        *,
        chunked: bool,
        chunk_size: int,
    ) -> None:
        blob_url = f"{client.prefix}://{container.get_blob_url(descriptor['digest'])}"
        if _request(client, container, blob_url, "HEAD").status_code == 200:
            return
        if chunked:
            response = _patch_blob(client, container, path, descriptor, chunk_size)
        else:
            response = _put_blob(client, container, path, descriptor)
        if _is_empty_blob(descriptor["digest"]) and response.status_code >= 300:
            # Like oras, accept that a registry refuses the empty blob, which pull
            # never fetches.
            return
        _check_response(response)

    @staticmethod
    def _get_manifest(client: Any, container: Any, reference: str) -> dict[str, Any]:
        """Fetch the manifest, verifying it against a digest reference when given."""
        expected = _reference_digest(reference)
        with _request(
            client,
            container,
            f"{client.prefix}://{container.manifest_url()}",
            "GET",
            headers={"Accept": OCI_MANIFEST_MEDIA_TYPE},
            stream=True,
        ) as response:
            _check_response(response)
            received = bytearray()
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                received += chunk
                if len(received) > _MAX_MANIFEST_SIZE:
                    raise InvalidModelPackError(
                        f"manifest is larger than {_MAX_MANIFEST_SIZE} bytes"
                    )
        payload = bytes(received)
        if expected:
            actual = _digest_bytes(payload, _algorithm(expected))
            if actual != expected:
                raise InvalidModelPackError(
                    f"manifest digest mismatch: expected {expected}, got {actual}"
                )
        _check_reported_digest(response, payload)
        try:
            manifest = json.loads(payload)
        except ValueError as error:
            raise InvalidModelPackError("manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise InvalidModelPackError("manifest must be a JSON object")
        return manifest

    @staticmethod
    def _download(
        client: Any, container: Any, descriptor: Mapping[str, Any], destination: Path
    ) -> None:
        """Stream a blob into ``destination`` and verify its size and digest."""
        with _open_blob(client, container, descriptor) as blob, destination.open(
            "wb"
        ) as output:
            shutil.copyfileobj(blob, output, _CHUNK_SIZE)
            blob.verify()

    @staticmethod
    def _build_config(
        config: ModelPackConfig | Mapping[str, Any] | str | Path,
        diff_ids: list[str],
    ) -> dict[str, Any]:
        if isinstance(config, ModelPackConfig):
            document = config.to_dict(diff_ids)
        elif isinstance(config, (str, Path)):
            with Path(config).expanduser().open(encoding="utf-8") as stream:
                try:
                    document = json.load(stream)
                except ValueError as error:
                    raise InvalidModelPackError(
                        f"ModelPack config file is not valid JSON: {config}"
                    ) from error
        else:
            document = deepcopy(dict(config))
        if not isinstance(document, dict):
            raise InvalidModelPackError("ModelPack config must be a JSON object")
        document["modelfs"] = {"type": "layers", "diffIds": list(diff_ids)}
        ModelPackClient._validate_config(document, len(diff_ids))
        return document

    @staticmethod
    def _validate_config(config: Any, layer_count: int) -> None:
        error = jsonschema.exceptions.best_match(_CONFIG_VALIDATOR.iter_errors(config))
        if error is not None:
            location = "/".join(str(part) for part in error.absolute_path) or "(root)"
            raise InvalidModelPackError(
                f"ModelPack config does not match the specification at {location}: "
                f"{error.message}"
            )
        diff_ids = config["modelfs"]["diffIds"]
        if len(diff_ids) != layer_count:
            raise InvalidModelPackError(
                "config.modelfs.diffIds must match the manifest layer count"
            )
        if not all(_is_digest(digest) for digest in diff_ids):
            raise InvalidModelPackError("all diffIds must be sha256 or sha512 digests")

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        if manifest.get("schemaVersion") != 2:
            raise InvalidModelPackError("manifest schemaVersion must be 2")
        if manifest.get("mediaType") != OCI_MANIFEST_MEDIA_TYPE:
            raise InvalidModelPackError("artifact is not an OCI image manifest")
        if manifest.get("artifactType") != MODEL_MANIFEST_ARTIFACT_TYPE:
            raise InvalidModelPackError("artifact is not a CNCF ModelPack")
        _validate_annotations(manifest.get("annotations"), "manifest")
        config = manifest.get("config")
        if (
            not isinstance(config, Mapping)
            or config.get("mediaType") != MODEL_CONFIG_MEDIA_TYPE
        ):
            raise InvalidModelPackError("manifest has no ModelPack config descriptor")
        _validate_descriptor(config, "manifest config")
        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise InvalidModelPackError("manifest.layers must be an array")
        for index, layer in enumerate(layers):
            if (
                not isinstance(layer, Mapping)
                or layer.get("mediaType") not in MODEL_LAYER_MEDIA_TYPES
            ):
                raise InvalidModelPackError("manifest contains a non-ModelPack layer")
            _validate_descriptor(layer, f"manifest layer {index}")


def _validate_descriptor(descriptor: Mapping[str, Any], label: str) -> None:
    if not _is_digest(descriptor.get("digest")):
        raise InvalidModelPackError(f"{label} digest must be a sha256 or sha512 digest")
    size = descriptor.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise InvalidModelPackError(f"{label} size must be a non-negative integer")
    _validate_annotations(descriptor.get("annotations"), label)


def _validate_annotations(annotations: Any, label: str) -> None:
    if annotations is None:
        return
    if not isinstance(annotations, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in annotations.items()
    ):
        raise InvalidModelPackError(f"{label} annotations must map strings to strings")


__all__ = ["ModelPackClient"]
