import streamlit as st
import pyarrow as pa
import seaborn as sns

from pybalance.propensity import PropensityScoreMatcher
from pybalance.sim import generate_toy_dataset
from pybalance.visualization import (
    plot_numeric_features,
    plot_categoric_features,
    plot_per_feature_loss,
)
from pybalance.utils import BALANCE_CALCULATORS

OBJECTIVES = list(BALANCE_CALCULATORS.keys())
OBJECTIVES.remove("base")

# Create a Streamlit app
# st.title('Propensity Score Matcher App')


# Create a sidebar for inputting parameters
st.sidebar.title("Parameters")
n_pool = st.sidebar.number_input(
    "Pool size",
    min_value=1,
    step=1000,
    value=10000,
    help="Number of patients in the pool population",
)
n_target = st.sidebar.number_input(
    "Target size",
    min_value=1,
    step=100,
    value=1000,
    help="Number of patients in the target population",
)
# seed = st.sidebar.number_input('Random Seed', min_value=0, step=1, value=45, help='Random seed for dataset generation')
seed = 45
objective = st.sidebar.selectbox("Objective", OBJECTIVES)
# caliper = st.sidebar.number_input('Caliper', min_value=0.0, max_value=1.0, value=1.0, step=0.01, help='If defined, restricts matches to those patients with propensity scores within the caliper of each other')
max_iter = st.sidebar.number_input(
    "Max Iterations",
    min_value=1,
    help="Maximum number of hyperparameters to try before returning the best match",
)
time_limit = st.sidebar.number_input(
    "Time Limit",
    min_value=10,
    help="Restrict hyperparameter search based on time in seconds",
)
method = st.sidebar.selectbox("Method", ["greedy", "linear_sum_assignment"])
cumulative = st.sidebar.checkbox("Cumulative plots", value=False)
if cumulative:
    bins = 500
else:
    bins = 10

palette = sns.color_palette("colorblind")

# Update the parameters based on user input
matching_data = generate_toy_dataset(n_pool, n_target, seed)
pre_matching_data = matching_data.copy()
hue_order = list(matching_data.populations)

# Create a button to trigger the match() method
if st.sidebar.button("Match"):

    # Create an instance of PropensityScoreMatcher
    matcher = PropensityScoreMatcher(
        matching_data, objective, None, max_iter, time_limit, method
    )

    # Call the match() method
    post_matching_data = matcher.match()
    post_matching_data.data.loc[:, "population"] = (
        post_matching_data["population"] + " (postmatch)"
    )
    matching_data.append(post_matching_data.data)

balance_calculator = BALANCE_CALCULATORS[objective](pre_matching_data)
st.sidebar.write(balance_calculator.__doc__)
hue_order += list(set(matching_data.populations) - set(hue_order))

# Display the figures
if matching_data:

    tab1, tab2, tab3 = st.tabs(["Numeric", "Categoric", "SMD"])
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
        for i, col in enumerate(st.columns(len(matching_data.headers["categoric"]))):
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
