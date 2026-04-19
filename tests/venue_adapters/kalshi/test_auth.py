"""
Unit tests for the Kalshi RSA-PSS-SHA256 auth signer.

PSS signatures are probabilistic (random salt) so we cannot pin against a
fixed signature byte-string. We test:
  - the three KALSHI-ACCESS-* headers are present and non-empty
  - the signature validates under the matching PUBLIC key using the exact
    PSS parameters the signer used
  - timestamp is a millisecond-precision integer string
  - the path (not URL, not query string) is what gets signed
  - WebSocket handshake headers sign the WS path with GET
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from executor.venue_adapters.kalshi.auth import KalshiAuth


@pytest.fixture(scope="module")
def ephemeral_key(tmp_path_factory) -> tuple[Path, rsa.RSAPrivateKey]:
    """Generate a throwaway RSA key for the test — never touches the real one."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path_factory.mktemp("kalshi_auth") / "test.key"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path, key


def _verify(pub, timestamp_ms: str, method: str, path: str, sig_b64: str) -> None:
    message = f"{timestamp_ms}{method.upper()}{path}".encode()
    pub.verify(
        base64.b64decode(sig_b64),
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_headers_verify_against_matching_public_key(ephemeral_key):
    path, priv = ephemeral_key
    auth = KalshiAuth(api_key_id="test-key-id", private_key_path=str(path))
    headers = auth.headers("GET", "/trade-api/v2/markets")

    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["Content-Type"] == "application/json"
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    sig = headers["KALSHI-ACCESS-SIGNATURE"]
    assert ts.isdigit() and len(ts) >= 13
    assert len(sig) > 80  # base64 of a 2048-bit signature

    _verify(priv.public_key(), ts, "GET", "/trade-api/v2/markets", sig)


def test_query_string_is_stripped_from_signature_path(ephemeral_key):
    path, priv = ephemeral_key
    auth = KalshiAuth(api_key_id="k", private_key_path=str(path))

    # Sign twice: once with query, once without. Signatures cover only the path.
    ts = "1700000000000"
    _, sig_with_qs = auth.sign("GET", "/trade-api/v2/markets?status=open", timestamp_ms=ts)
    _, sig_no_qs = auth.sign("GET", "/trade-api/v2/markets", timestamp_ms=ts)
    # PSS is randomized so signatures differ byte-wise even with identical inputs;
    # we instead verify both validate against the bare path.
    _verify(priv.public_key(), ts, "GET", "/trade-api/v2/markets", sig_with_qs)
    _verify(priv.public_key(), ts, "GET", "/trade-api/v2/markets", sig_no_qs)


def test_ws_headers_sign_get_on_ws_path(ephemeral_key):
    path, priv = ephemeral_key
    auth = KalshiAuth(api_key_id="k", private_key_path=str(path))
    h = auth.ws_headers("/trade-api/ws/v2")
    _verify(
        priv.public_key(),
        h["KALSHI-ACCESS-TIMESTAMP"],
        "GET",
        "/trade-api/ws/v2",
        h["KALSHI-ACCESS-SIGNATURE"],
    )


def test_timestamp_is_recent_ms(ephemeral_key):
    path, _ = ephemeral_key
    auth = KalshiAuth(api_key_id="k", private_key_path=str(path))
    h = auth.headers("GET", "/trade-api/v2/markets")
    ts = int(h["KALSHI-ACCESS-TIMESTAMP"])
    now_ms = int(time.time() * 1000)
    assert abs(now_ms - ts) < 5_000  # within 5s


def test_auth_from_env_requires_both_vars(ephemeral_key, monkeypatch):
    path, _ = ephemeral_key
    from executor.venue_adapters.kalshi.auth import auth_from_env

    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="KALSHI_API_KEY_ID"):
        auth_from_env()

    monkeypatch.setenv("KALSHI_API_KEY_ID", "kid")
    with pytest.raises(RuntimeError, match="KALSHI_PRIVATE_KEY_PATH"):
        auth_from_env()

    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(path))
    auth = auth_from_env()
    assert auth.api_key_id == "kid"
