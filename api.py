from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bin.main import match, generate_data
from dotenv import load_dotenv
import os
import pandas as pd
import json

load_dotenv()

fastapi_server = os.getenv("FASTAPI_SERVER", "localhost")
fastapi_port = os.getenv("FASTAPI_PORT", "8000")
app = FastAPI(
    title="PyBalance API",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class GenerateDataRequest(BaseModel):
    n_pool: int
    n_target: int


class MatchRequest(BaseModel):
    matching_data: Dict  # Replace with proper type
    objective: str
    max_iter: int = Field(100)


@app.post("/generate_data")
async def generate_data_endpoint(request: GenerateDataRequest):
    matching_data = generate_data(request.n_pool, request.n_target)
    return matching_data


@app.post("/match")
async def match_endpoint(request: MatchRequest):
    matching_data_dict = request.matching_data
    post_matching_data = match(matching_data_dict, request.objective, request.max_iter)
    print(
        f"post_matching_datapost_matching_datapost_matching_datapost_matching_data {post_matching_data}"
    )
    return {"post_matching_data": post_matching_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=fastapi_server, port=fastapi_port)
