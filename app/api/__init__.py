# api/__init__.py
from fastapi import FastAPI

from . import routes_home, routes_classify, routes_chat, routes_health
#from . import routes_query

def init_routers(app: FastAPI) -> None:
    """Include all API routers into the main app."""
    app.include_router(routes_home.router)
    app.include_router(routes_classify.router)
    app.include_router(routes_chat.router)
    #app.include_router(routes_query.router)
    app.include_router(routes_health.router)