import os
import ssl

import certifi


def get_config():
    return {
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL") or "qwen3-embedding:0.6b",
        "CHAT_MODEL": os.getenv("CHAT_MODEL"),
        "INDEX_NAME": os.getenv("INDEX_NAME"),
    }


def set_ssl_certificates():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    return ssl_context
