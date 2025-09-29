from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from wal_fact_checker.a2a.app import a2a_app

app: FastAPI = get_fast_api_app(
    agents_dir="./src",
    web=True,
)

app.router.routes += a2a_app.router.routes
