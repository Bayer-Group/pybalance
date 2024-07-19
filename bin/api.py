from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from pybalance.propensity import PropensityScoreMatcher
from pybalance.sim import generate_toy_dataset
from pybalance.visualization import (
    plot_numeric_features,
    plot_categoric_features,
    plot_per_feature_loss,
)
from pybalance.utils import BALANCE_CALCULATORS, split_target_pool, MatchingData

from dotenv import load_dotenv

load_dotenv()

fastapi_server = os.getenv("FASTAPI_SERVER", "localhost")
fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
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


def match(payload: Dict, objective: str, max_iter: int = 100):
    print("Inside Matching data...")
    matching_data_recreated = MatchingData.from_dict(payload)
    # Create an instance of PropensityScoreMatcher
    method = "greedy"
    matcher = PropensityScoreMatcher(
        matching_data_recreated, objective, None, max_iter, 10, method
    )
    print(f"matcher {matcher}")
    # Call the match() method
    post_matching_data = matcher.match()
    post_matching_data.data.loc[:, "population"] = (
        post_matching_data["population"] + " (postmatch)"
    )
    print(f"post_matching_data {post_matching_data}")
    return post_matching_data.to_dict()


def generate_data(n_pool: int, n_target: int):
    print("Inside Generating data...")
    seed = 45
    # n_pool, n_target = st.session_state["n_pool"], st.session_state["n_target"]
    matching_data = generate_toy_dataset(n_pool, n_target, seed)

    # st.session_state["first_run"] = False
    print(matching_data.head(5))
    return matching_data.to_dict()


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
    return {"post_matching_data": post_matching_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=fastapi_server, port=fastapi_port)
