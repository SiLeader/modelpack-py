"""Tests against a local HTTP registry that uses Docker-style token auth."""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import re
import secrets
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import oras.auth

from modelpack import (
    InvalidModelPackError,
    ModelLayer,
    ModelPackClient,
    ModelPackConfig,
)
from modelpack.constants import OCI_MANIFEST_MEDIA_TYPE

ROUTE = re.compile(r"/v2/(?P<name>.+?)/(?P<kind>blobs/uploads|blobs|manifests)/(?P<ref>[^/]*)")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _basic(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


class RegistryHandler(BaseHTTPRequestHandler):
    server: RegistryServer
    # A client that announces a body and does not send it must not hang the test.
    timeout = 5

    def log_message(self, format, *args) -> None:
        pass

    def do_GET(self) -> None:
        self._handle()

    do_HEAD = do_POST = do_PUT = do_PATCH = do_GET

    def _handle(self) -> None:
        registry = self.server
        url = urlsplit(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        authorization = self.headers.get("Authorization")
        with registry.lock:
            registry.requests.append((self.command, url.path))
            if authorization:
                registry.authorizations.append(authorization)

        if url.path == "/token":
            return self._issue_token(parse_qs(url.query), authorization)
        match = ROUTE.fullmatch(url.path)
        if match is None:
            return self._send(404)
        name, kind, ref = match["name"], match["kind"], match["ref"]
        actions = ("pull",) if self.command in ("GET", "HEAD") else ("pull", "push")
        if kind == "blobs/uploads" and self.command == "PUT" and registry.expire_tokens_before_put:
            registry.expire_tokens_before_put = False
            registry.tokens.clear()
        if not self._authorized(authorization, name, actions):
            return self._challenge(name, actions)

        if kind == "blobs":
            if ref not in registry.blobs:
                return self._send(404)
            return self._send(200, registry.blobs[ref])
        if kind == "blobs/uploads":
            return self._upload(name, ref, url.query, body)
        return self._manifest(name, ref, body)

    def _issue_token(self, query, authorization) -> None:
        registry = self.server
        granted = set()
        for scope in query.get("scope", []):
            name, _, actions = scope.partition(":")[2].rpartition(":")
            for action in actions.split(","):
                if action == "pull" or authorization == registry.basic_authorization:
                    granted.add((name, action))
        token = secrets.token_hex(16)
        with registry.lock:
            registry.tokens[token] = granted
            registry.issued_tokens.append(token)
        self._send(200, json.dumps({"token": token}).encode())

    def _authorized(self, authorization, name, actions) -> bool:
        if self.server.anonymous:
            return True
        if not authorization or not authorization.startswith("Bearer "):
            return False
        granted = self.server.tokens.get(authorization.removeprefix("Bearer "), set())
        return all((name, action) in granted for action in actions)

    def _challenge(self, name, actions) -> None:
        realm = f"http://{self.server.host}/token"
        scope = f"repository:{name}:{','.join(actions)}"
        header = f'Bearer realm="{realm}",service="test-registry",scope="{scope}"'
        self._send(401, b'{"errors":[{"code":"UNAUTHORIZED"}]}', {"WWW-Authenticate": header})

    def _upload(self, name, ref, query, body) -> None:
        registry = self.server
        uploads = registry.uploads
        if self.command == "POST":
            upload = secrets.token_hex(8)
            location = f"/v2/{name}/blobs/uploads/{upload}"
            if registry.upload_server is not None:
                registry.upload_server.uploads[upload] = bytearray()
                return self._send(
                    202, headers={"Location": f"http://{registry.upload_server.host}{location}"}
                )
            uploads[upload] = bytearray()
            return self._send(202, headers={"Location": location})
        target = registry.redirect_puts_to
        if self.command == "PUT" and target is not None and "redirected" not in parse_qs(query):
            target.uploads.setdefault(ref, bytearray())
            location = f"http://{target.host}{self.path}&redirected=1"
            return self._send(307, headers={"Location": location})
        uploads[ref] += body
        if self.command == "PATCH":
            self.server.patches += 1
            return self._send(202, headers={"Location": f"/v2/{name}/blobs/uploads/{ref}"})
        content = bytes(uploads.pop(ref))
        digest = parse_qs(query)["digest"][0]
        if _sha256(content) != digest:
            return self._send(400, b'{"errors":[{"code":"DIGEST_INVALID"}]}')
        self.server.blobs[digest] = content
        self._send(201, headers={"Docker-Content-Digest": digest})

    def _manifest(self, name, ref, body) -> None:
        manifests = self.server.manifests
        if self.command == "PUT":
            digest = _sha256(body)
            manifests[(name, ref)] = manifests[(name, digest)] = body
            return self._send(201, headers={"Docker-Content-Digest": digest})
        if (name, ref) not in manifests:
            return self._send(404)
        payload = manifests[(name, ref)]
        headers = {"Content-Type": OCI_MANIFEST_MEDIA_TYPE, "Docker-Content-Digest": _sha256(payload)}
        self._send(200, payload, headers)

    def _send(self, status, body=b"", headers=None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class RegistryServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        super().__init__(("127.0.0.1", 0), RegistryHandler)
        self.host = f"127.0.0.1:{self.server_address[1]}"
        self.basic_authorization = _basic(username, password) if username else None
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str]] = []
        self.authorizations: list[str] = []
        self.issued_tokens: list[str] = []
        self.tokens: dict[str, set[tuple[str, str]]] = {}
        self.blobs: dict[str, bytes] = {}
        self.uploads: dict[str, bytearray] = {}
        self.manifests: dict[tuple[str, str], bytes] = {}
        self.patches = 0
        self.expire_tokens_before_put = False
        self.anonymous = False
        self.upload_server: RegistryServer | None = None
        self.redirect_puts_to: RegistryServer | None = None
        threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()

    def close(self) -> None:
        self.shutdown()
        self.server_close()


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # oras reads and writes ~/.docker/config.json.
        home = mock.patch.dict("os.environ", {"HOME": str(self.root / "home")})
        home.start()
        self.addCleanup(home.stop)
        self.model = self.root / "model.bin"
        self.model.write_bytes(b"weights" * 1000)

    def registry(self, username=None, password=None) -> RegistryServer:
        server = RegistryServer(username, password)
        self.addCleanup(server.close)
        return server

    def test_push_and_pull_with_token_auth(self):
        registry = self.registry("user", "secret")
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")

        client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())
        output = self.root / "output"
        client.pull(f"{registry.host}/models/tiny:1", output)

        self.assertEqual((output / "model.bin").read_bytes(), self.model.read_bytes())
        self.assertEqual(registry.patches, 0)

    def test_credentials_and_tokens_are_not_sent_to_another_registry(self):
        registry = self.registry("user", "secret")
        other = self.registry()
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")
        # First with only the Basic credentials, then with a cached bearer token.
        with self.assertRaises(ValueError):
            client.pull(f"{other.host}/models/tiny:1", self.root / "output")
        client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())
        with self.assertRaises(ValueError):
            client.pull(f"{other.host}/models/tiny:1", self.root / "output")

        self.assertTrue(other.authorizations)
        self.assertNotIn(registry.basic_authorization, other.authorizations)
        leaked = {f"Bearer {token}" for token in registry.issued_tokens}
        self.assertFalse(leaked & set(other.authorizations))

    def test_token_for_one_repository_is_replaced_for_another(self):
        registry = self.registry("user", "secret")
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")
        client.push(f"{registry.host}/models/a:1", [self.model], ModelPackConfig())
        client.pull(f"{registry.host}/models/a:1", self.root / "a")

        other_model = self.root / "other.bin"
        other_model.write_bytes(b"other weights")
        client.push(f"{registry.host}/models/b:1", [other_model], ModelPackConfig())

        client.pull(f"{registry.host}/models/b:1", self.root / "b")
        self.assertEqual((self.root / "b/other.bin").read_bytes(), b"other weights")

    def test_streamed_put_is_resent_in_full_after_a_challenge(self):
        registry = self.registry("user", "secret")
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")
        registry.expire_tokens_before_put = True

        client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())

        self.assertIn(_sha256(self.model.read_bytes()), registry.blobs)
        self.assertFalse(registry.expire_tokens_before_put)

    def test_chunked_upload_is_opt_in(self):
        registry = self.registry("user", "secret")
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")

        client.push(
            f"{registry.host}/models/tiny:1",
            [self.model],
            ModelPackConfig(),
            chunked=True,
            chunk_size=1024,
        )

        self.assertGreater(registry.patches, 1)
        self.assertIn(_sha256(self.model.read_bytes()), registry.blobs)

    def write_credentials(self, path: Path, host: str, password: str) -> Path:
        auth = base64.b64encode(f"user:{password}".encode()).decode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"auths": {host: {"auth": auth}}}))
        return path

    def test_config_path_is_read_on_every_call(self):
        registry = self.registry("user", "secret")
        config = self.write_credentials(self.root / "config.json", registry.host, "nope")
        client = ModelPackClient(insecure=True)
        reference = f"{registry.host}/models/tiny:1"

        with self.assertRaises(ValueError):
            client.push(reference, [self.model], ModelPackConfig(), config_path=config)
        self.write_credentials(config, registry.host, "secret")
        client.push(reference, [self.model], ModelPackConfig(), config_path=config)

        self.assertIn(("models/tiny", "1"), registry.manifests)

    def test_config_path_takes_precedence_over_docker_config(self):
        registry = self.registry("user", "secret")
        reference = f"{registry.host}/models/tiny:1"
        docker_config = self.root / "home/.docker/config.json"
        cases = {"custom is right": ("nope", "secret"), "custom is wrong": ("secret", "nope")}
        # Merging the files in set order would pick either file, depending on the names.
        for (name, (docker_password, custom_password)), index in itertools.product(
            cases.items(), range(8)
        ):
            with self.subTest(name, index=index):
                custom_config = self.root / f"custom-{index}.json"
                self.write_credentials(docker_config, registry.host, docker_password)
                self.write_credentials(custom_config, registry.host, custom_password)
                client = ModelPackClient(insecure=True)
                if custom_password == "secret":
                    client.push(reference, [self.model], ModelPackConfig(), config_path=custom_config)
                else:
                    with self.assertRaises(ValueError):
                        client.push(reference, [self.model], ModelPackConfig(), config_path=custom_config)

    def test_logout_removes_the_credentials_login_saved(self):
        registry = self.registry("user", "secret")
        reference = f"{registry.host}/models/tiny:1"
        cases = {"docker config": None, "custom config": self.root / "custom.json"}
        for name, config_path in cases.items():
            with self.subTest(name):
                client = ModelPackClient(insecure=True)
                client.login(registry.host, "user", "secret", config_path=config_path)
                client.push(reference, [self.model], ModelPackConfig(), config_path=config_path)

                client.logout(registry.host, config_path=config_path)

                saved = json.loads((config_path or self.root / "home/.docker/config.json").read_text())
                self.assertNotIn(registry.host, saved["auths"])
                for pusher in (client, ModelPackClient(insecure=True)):
                    with self.assertRaises(ValueError):
                        pusher.push(reference, [self.model], ModelPackConfig(), config_path=config_path)

    def test_streamed_put_is_resent_in_full_after_a_redirect(self):
        registry = self.registry("user", "secret")
        registry.redirect_puts_to = registry
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")

        client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())

        self.assertIn(_sha256(self.model.read_bytes()), registry.blobs)

    def test_credentials_are_not_sent_to_another_upload_host(self):
        for redirect in (False, True):
            with self.subTest(redirect=redirect):
                registry = self.registry("user", "secret")
                storage = self.registry("user", "secret")
                if redirect:
                    registry.redirect_puts_to = storage
                else:
                    registry.upload_server = storage
                client = ModelPackClient(insecure=True)
                client.login(registry.host, "user", "secret")

                with self.assertRaises((ValueError, oras.auth.AuthenticationException)):
                    client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())

                self.assertIn("PUT", [method for method, _ in storage.requests])
                self.assertEqual(storage.authorizations, [])

    def test_upload_to_another_host_works_without_credentials(self):
        for redirect in (False, True):
            with self.subTest(redirect=redirect):
                registry = self.registry("user", "secret")
                storage = self.registry()
                storage.anonymous = True
                if redirect:
                    registry.redirect_puts_to = storage
                else:
                    registry.upload_server = storage
                client = ModelPackClient(insecure=True)
                client.login(registry.host, "user", "secret")

                client.push(f"{registry.host}/models/tiny:1", [self.model], ModelPackConfig())

                self.assertIn(_sha256(self.model.read_bytes()), storage.blobs)
                self.assertEqual(storage.authorizations, [])

    def test_pull_by_digest_detects_a_tampered_manifest(self):
        registry = self.registry("user", "secret")
        client = ModelPackClient(insecure=True)
        client.login(registry.host, "user", "secret")
        pushed = client.push(f"{registry.host}/models/tiny:1", [ModelLayer(self.model)], ModelPackConfig())
        reference = f"{registry.host}/models/tiny@{pushed.digest}"
        client.pull(reference, self.root / "output")

        key = ("models/tiny", pushed.digest)
        registry.manifests[key] = registry.manifests[key].replace(b"model.bin", b"evil.bin")
        with self.assertRaisesRegex(InvalidModelPackError, "manifest digest mismatch"):
            client.pull(reference, self.root / "tampered")


if __name__ == "__main__":
    unittest.main()
