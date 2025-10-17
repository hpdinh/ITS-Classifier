import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from services import ClassifierService, LLMClient, SnowAssignmentGroupsRetriever, SnowTicketQuery
from api import init_routers


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger()


# ---------------- FastAPI ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize classifier
        app.state.classifier = ClassifierService()
        logger.info("ClassifierService initialized")

        # Initialize LLM client
        app.state.llm_client = LLMClient()
        logger.info("LLMClient initialized")

        # Initialize ServiceNow assignment groups retriever
        app.state.groups_retriever = SnowAssignmentGroupsRetriever()
        logger.info("SnowAssignmentGroupsRetriever initialized")

        # Initialize ticket query (depends on LLM)
        app.state.ticket_query = SnowTicketQuery(llm_client=app.state.llm_client)
        logger.info("SnowTicketQuery initialized")

        app.state.running = True
        logger.info("Application started successfully")

    except Exception as e:
        app.state.running = False
        logger.exception("Application failed to initialize")
        raise e

    yield

    # Shutdown
    logger.info("Shutting down app...")

    # Close HTTP client
    if hasattr(app.state, "llm_client"):
        try:
            await app.state.llm_client.aclose()
            logger.info("Closed LLM client")
        except Exception as e:
            logger.warning(f"Error closing LLM client: {e}")
    # Close classifier HTTP
    if hasattr(app.state, "classifier") and app.state.classifier.use_api:
        try:
            await app.state.classifier.aclose()
            logger.info("Closed Classifier API")
        except Exception as e:
            logger.warning(f"Error closing classifier client: {e}")
    # Close Redis if using it
    try:
        from services.session_service import SessionState
        if SessionState._settings.use_redis and hasattr(SessionState, "_redis_client"):
            SessionState._redis_client.close()
            logger.info("Closed Redis client")
    except Exception as e:
        logger.warning(f"Error closing Redis client: {e}")


# ---------------- App Factory ----------------
app = FastAPI(lifespan=lifespan)

# Templates & static assets
templates = Jinja2Templates(directory="assets/templates")
app.mount("/static", StaticFiles(directory="assets/static"), name="static")

# Routers
init_routers(app)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
