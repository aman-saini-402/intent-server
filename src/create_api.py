import os
from datetime import datetime

import yaml
from fastapi import FastAPI, HTTPException, status, APIRouter, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware

from .prediction_pipeline import Pipeline

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)


class CreateAPI:
    """
    Create an API for LLM inference
    """

    def __init__(self):
        """
        Load and initialize intent classification pipelines
        """
        print("\nLoading Intent Classifier...")
        self.ic = Pipeline()
        print("Classifier Loaded Successfully!")

    def _authenticate_user(self, token: str) -> None:
        secure_token = os.environ["BEARER_TOKEN"]
        if token != secure_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )

    def get_api(self) -> FastAPI:
        """
        Create a FastAPI wrapper around intent classifiers
        """
        # Create an API
        app = FastAPI(
            title="Intent Classifier",
            openai_url="/openai.json"
        )

        origins = ["http://localhost:3000"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        api_router = APIRouter()

        # Custom exception handler
        @app.exception_handler(HTTPException)
        async def custom_exception(request, exc):
            error_message = dict(
                status=exc.status_code,
                timestamp=datetime.now().isoformat(),
                message="An error occurred!",
                error=exc.detail
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=error_message
            )

        # Update version at openapi endpoint
        def custom_openapi():
            if app.openapi_schema:
                app.openapi_schema["info"]["version"] = config["API_VERSION"]
                return app.openapi_schema
            openapi_schema = get_openapi(
                title=app.title,
                version=config["API_VERSION"],
                description=app.description,
                routes=app.routes,
            )
            app.openapi_schema = openapi_schema
            return app.openapi_schema

        app.openapi = custom_openapi

        @api_router.get("/intent-server/chatgpt/p5")
        async def read_query(
                access_token: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
                prompt: str = "",
                prompt2: str = "",
                followup_q: str = ""
        ) -> dict:
            # Authenticate user
            self._authenticate_user(access_token.credentials)

            # Call Intent Classifier
            llm_response = await self.ic.run(prompt, prompt2, followup_q)

            return llm_response

        app.include_router(api_router)

        return app
