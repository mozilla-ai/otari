from fastapi import FastAPI

from gateway.core.config import load_config
from gateway.main import create_app


def create_dev_app() -> FastAPI:
    return create_app(load_config())
