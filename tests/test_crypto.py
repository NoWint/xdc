from __future__ import annotations



from xdc.crypto import (
    NodeIdentity,
    NonceCache,
    generate_tls_material,
    node_id_from_public_key,
    sign_request,
    verify_node_signature,
    verify_signed_request,
)


def test_identity_is_stable_and_signatures_verify(tmp_path):
    ident = NodeIdentity.load_or_create(tmp_path)
    again = NodeIdentity.load_or_create(tmp_path)
    assert ident.node_id == again.node_id
    assert node_id_from_public_key(ident.public_key) == ident.node_id
    payload = b"hello xdc"
    sig = ident.sign(payload)
    assert ident.verify(payload, sig)
    assert not ident.verify(payload + b"x", sig)
    assert verify_node_signature(ident.public_key, payload, sig)


def test_signed_request_and_replay(tmp_path):
    ident = NodeIdentity.load_or_create(tmp_path)
    headers = sign_request(ident, "POST", "/v1/x?a=1", b'{"b":2}')
    nonces = NonceCache()
    ok, reason = verify_signed_request(ident.public_key, "POST", "/v1/x?a=1", b'{"b":2}', headers, nonces)
    assert ok, reason
    ok, reason = verify_signed_request(ident.public_key, "POST", "/v1/x?a=1", b'{"b":2}', headers, nonces)
    assert not ok and "replay" in reason
    # Wrong body is a signature failure with a fresh nonce.
    h2 = sign_request(ident, "POST", "/v1/x?a=1", b'{"b":2}')
    ok, reason = verify_signed_request(ident.public_key, "POST", "/v1/x?a=1", b'{"b":3}', h2, NonceCache())
    assert not ok and "signature" in reason


def test_tls_material(tmp_path):
    paths = generate_tls_material(tmp_path, ["127.0.0.1", "localhost"])
    for key in ("ca", "cert", "key"):
        assert (tmp_path / ({"ca": "ca.pem", "cert": "server.crt", "key": "server.key"}[key])).exists()
    from cryptography import x509

    cert = x509.load_pem_x509_certificate(open(paths["cert"], "rb").read())
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in names.get_values_for_type(x509.DNSName)
