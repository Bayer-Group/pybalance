import os
import sys
from dotenv import load_dotenv

# Assuming 'pybalance' directory is at the same level as 'streamlit_app.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict

import streamlit as st
import pandas as pd
import seaborn as sns

from pybalance.propensity import PropensityScoreMatcher
from pybalance.sim import generate_toy_dataset
from pybalance.visualization import (
    plot_numeric_features,
    plot_categoric_features,
    plot_per_feature_loss,
)
from pybalance.utils import BALANCE_CALCULATORS, split_target_pool, MatchingData
import requests


# Load environment variables from .env file
load_dotenv()

# Read server and port from environment variables
fastapi_server = os.getenv("FASTAPI_SERVER", "localhost")
fastapi_port = os.getenv("FASTAPI_PORT", "8000")
FASTAPI_URL = f"http://{fastapi_server}:{fastapi_port}"

OBJECTIVES = list(BALANCE_CALCULATORS.keys())
OBJECTIVES.remove("base")

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("logo.png", width=400)
st.markdown(
    "<h5> <center>The matching toolkit for observational data.</center></h5>",
    unsafe_allow_html=True,
)
placeholder = st.empty()


def generate_data(n_pool: int, n_target: int):
    print("Inside Generating data...")
    seed = 45
    # n_pool, n_target = st.session_state["n_pool"], st.session_state["n_target"]
    matching_data = generate_toy_dataset(n_pool, n_target, seed)

    # st.session_state["first_run"] = False
    print(matching_data.head(5))
    return matching_data.to_dict()


def load_data():
    uploaded_file = st.session_state.get("uploaded_file")
    if uploaded_file is None:
        st.warning(f"Please specify a file to upload.")
        return
    population_column = st.session_state.get("population_column")
    try:
        df = pd.read_csv(uploaded_file)
    except:  # FIXME catch specific exception UnicodeDecodeError
        df = pd.read_parquet(uploaded_file)
    matching_data = MatchingData(
        df,
        population_col=population_column,
    )
    try:
        split_target_pool(matching_data)
    except:
        st.warning(
            f"Cannot split data based on {population_column}. Please specify a split column with two unique values."
        )
        return

    st.session_state["first_run"] = False
    st.session_state["matching_data"] = matching_data


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
    st.session_state["post_matching_data"] = post_matching_data
    print(f"post_matching_data {post_matching_data}")
    return post_matching_data.to_dict()


def load_front_page():
    st.markdown("<h5>Generate a simulated dataset</h5>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        n_pool = st.number_input(
            "Pool size",
            min_value=1,
            step=1000,
            value=10000,
            key="n_pool",
            help="Number of patients in the pool (by convention, larger) population",
        )
    with col2:
        n_target = st.number_input(
            "Target size",
            min_value=1,
            step=100,
            value=1000,
            key="n_target",
            help="Number of patients in the target (by convention, smaller) population",
        )

    if st.button("Generate"):
        # Prepare the payload
        payload = {"n_pool": n_pool, "n_target": n_target}

        # Call the FastAPI endpoint
        response = requests.post(f"{FASTAPI_URL}/generate_data", json=payload)

        if response.status_code == 200:
            matching_data_json = response.json()
            st.session_state["matching_data"] = matching_data_json
            # matching_instance = MatchingData.from_json(matching_data_json)
            st.session_state["first_run"] = False
            st.success("Data generated successfully")
            st.rerun()  # Force the script to rerun
        else:
            st.error("Failed to generate data")

    st.write("---")
    st.markdown("<h5>Upload your own data</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.file_uploader(
            "Upload file",
            type=["csv", "parquet"],
            help="Choose a matching file to upload.",
            key="uploaded_file",
        )
    with col2:
        st.text_input(
            "population column",
            value="population",
            key="population_column",
            help="Column name for splitting data into different populations",
        )
    st.button("Submit", on_click=load_data)
    st.write(
        "We retain your data only for the duration of your user session. You are responsible for the security of your data and compliance with your data license."
    )


# Load landing page
if st.session_state.get("first_run", True):
    with placeholder.container():
        load_front_page()

if not st.session_state.get("first_run", True):
    with placeholder.container():

        matching_data = st.session_state.get("matching_data").copy()
        matching_data = MatchingData.from_dict(matching_data)
        target, pool = split_target_pool(matching_data)

        # Create a sidebar for inputting parameters
        st.sidebar.title("Matching parameters")
        st.sidebar.number_input(
            "Matched pool size",
            min_value=1,
            step=100,
            value=len(target),
            max_value=len(pool),
            key="n_pool_matched",
            help="Number of patients in the pool population after matching",
        )
        st.sidebar.number_input(
            "Matched Target size",
            min_value=1,
            step=100,
            value=len(target),
            max_value=len(target),
            key="n_target_matched",
            help="Number of patients in the target population after matching",
        )
        # seed = st.sidebar.number_input('Random Seed', min_value=0, step=1, value=45, help='Random seed for dataset generation')
        objective = st.sidebar.selectbox("Objective", OBJECTIVES, key="objective")
        # caliper = st.sidebar.number_input('Caliper', min_value=0.0, max_value=1.0, value=1.0, step=0.01, help='If defined, restricts matches to those patients with propensity scores within the caliper of each other')
        st.sidebar.number_input(
            "Max Iterations",
            min_value=1,
            key="max_iter",
            help="Maximum number of hyperparameters to try before returning the best match",
        )
        time_limit = st.sidebar.number_input(
            "Time Limit",
            min_value=10,
            key="time_limit",
            help="Restrict hyperparameter search based on time in seconds",
        )
        # method = st.sidebar.selectbox("Method", ["greedy", "linear_sum_assignment"])
        cumulative = st.sidebar.checkbox("Cumulative plots", value=False)
        if cumulative:
            bins = 500
        else:
            bins = 10

        palette = sns.color_palette("colorblind")

        # Update the parameters based on user input
        hue_order = list(matching_data.populations)

        # Create a button to trigger the match() method
        if st.sidebar.button("Match"):
            if "matching_data" in st.session_state:
                matching_data_str = st.session_state["matching_data"]
                payload = {
                    "matching_data": matching_data_str,
                    "objective": st.session_state.get("objective"),
                    "max_iter": st.session_state.get("max_iter"),
                }

                # Call the FastAPI endpoint
                response = requests.post(f"{FASTAPI_URL}/match", json=payload)

                if response.status_code == 200:
                    post_matching_data_json = response.json().get("post_matching_data")
                    post_matching_instance = MatchingData.from_json(
                        post_matching_data_json
                    )
                    st.session_state["post_matching_instance"] = post_matching_instance
                    st.success("Data matched successfully")
                else:
                    st.error("Failed to match data")
            else:
                st.error("No matching data found. Generate data first.")
        # st.sidebar.button("Match", on_click=match)

        balance_calculator = BALANCE_CALCULATORS[objective](matching_data)
        st.sidebar.write(balance_calculator.__doc__)

        if "post_matching_data" in st.session_state.keys():
            post_matching_data = st.session_state["post_matching_data"]
            matching_data.append(post_matching_data.data)

        hue_order += list(set(matching_data.populations) - set(hue_order))
        print("populations", hue_order)
        tab1, tab2, tab3, tab4 = st.tabs(["Numeric", "Categoric", "SMD", "Summary"])
        with tab1:

            plot_vars = []
            for i, col in enumerate(st.columns(len(matching_data.headers["numeric"]))):
                with col:
                    col_name = matching_data.headers["numeric"][i]
                    if st.checkbox(col_name, value=True):
                        plot_vars.append(col_name)
            print("streamlit", plot_vars)
            numeric_fig = plot_numeric_features(
                matching_data,
                col_wrap=2,
                height=6,
                hue_order=hue_order,
                cumulative=cumulative,
                bins=bins,
                include_only=plot_vars,
                # palette=palette,
            )
            st.pyplot(numeric_fig)
            st.write("---")
            # import pdb
            # pdb.set_trace()
            summary = matching_data.describe_numeric().astype("object")
            summary = summary[summary.index.get_level_values(0).isin(plot_vars)]
            st.dataframe(summary, use_container_width=True)

        with tab2:
            plot_vars = []
            for i, col in enumerate(
                st.columns(len(matching_data.headers["categoric"]))
            ):
                with col:
                    col_name = matching_data.headers["categoric"][i]
                    if st.checkbox(col_name, value=True):
                        plot_vars.append(col_name)

            print("streamlit", plot_vars)
            categoric_fig = plot_categoric_features(
                matching_data,
                col_wrap=2,
                height=6,
                include_binary=True,
                hue_order=hue_order,
                include_only=plot_vars,
                # palette=palette,
            )
            st.pyplot(categoric_fig)
            st.write("---")
            summary = matching_data.describe_categoric().astype("object")
            summary = summary[summary.index.get_level_values(0).isin(plot_vars)]
            st.dataframe(summary, use_container_width=True)

        with tab3:
            categoric_fig = plot_per_feature_loss(
                matching_data,
                balance_calculator,
                hue_order=hue_order,
                debin=False,
                # palette=palette,
            )
            st.pyplot(categoric_fig)

        with tab4:
            st.markdown(matching_data._repr_html_(), unsafe_allow_html=True)
            st.dataframe(matching_data.counts(), use_container_width=True)
