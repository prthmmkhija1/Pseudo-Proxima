"""
Proxima Web API.

RESTful API for quantum simulation orchestration using FastAPI.
"""

from proxima.api.main import create_app, get_app

__all__ = [
    "create_app",
    "get_app",
]
