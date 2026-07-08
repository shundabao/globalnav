"""WSGI entrypoint for deploying GLOBALNAV."""

from globe_nav.config import DEFAULT_MODEL
from globe_nav.gui.app import create_app


app = create_app(model=DEFAULT_MODEL)
