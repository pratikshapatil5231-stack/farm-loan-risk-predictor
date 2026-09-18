import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Farm Loan Risk Predictor",
    page_icon="🌾",
    layout="wide"
)

pio.templates.default = "plotly_white"


# =========================================================
# LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_model_files():

    with open("saved_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    try:
        with open("target_encoder.pkl", "rb") as f:
            target_mapping = pickle.load(f)
    except Exception:
        target_mapping = {
            0: "High Risk ❌",
            1: "Medium Risk ⚠️",
            2: "Low Risk ✅"
        }

    return model, scaler, target_mapping


try:
    model, scaler, target_mapping = load_model_files()

except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.info(
        "Make sure saved_model.pkl, scaler.pkl and "
        "target_encoder.pkl are in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()


# =========================================================
# FEATURE NAMES
# =========================================================
try:
    expected_features = list(scaler.feature_names_in_)

except AttributeError:
    expected_features = [
        "age",
        "land_size",
        "income",
        "crop_type",
        "loan_amount",
        "loan_term",
        "previous_defaults",
        "rainfall",
        "soil_type",
        "market_index"
    ]


# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "current_crop" not in st.session_state:
    st.session_state.current_crop = None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
menu = st.sidebar.radio(
    "📍 Navigate",
    [
        "🏠 Home",
        "📊 Predict Risk",
        "📈 Insights",
        "📊 Visuals",
        "📋 History"
    ],
    key="main_navigation"
)


# =========================================================
# HOME
# =========================================================
if menu == "🏠 Home":

    st.title("🌾 AI-Powered Farm Loan Risk Prediction System")

    st.markdown("""
    Welcome to the **AI Farm Loan Risk Prediction System**.

    This application uses a machine learning model to estimate
    farm loan risk categories from farmer and agricultural information.

    ## 🎯 Key Features

    - 🔮 Real-time loan risk prediction
    - 📊 Risk probability and confidence score
    - 📈 Agricultural market insights
    - 📋 Prediction history
    - 💾 Export prediction history
    - 📊 Interactive visualizations

    ## 🛠️ Technology Stack

    - **Frontend:** Streamlit
    - **Machine Learning:** Scikit-learn
    - **Visualization:** Plotly
    - **Data:** Farm loan dataset

    ---

    ### 🚜 How to use

    1. Open **Predict Risk**
    2. Enter farmer details
    3. Click **Predict Risk Now**
    4. View the predicted risk
    5. Check probability and feature importance
    6. Explore **Insights**, **Visuals**, and **History**
    """)

    st.success("🌾 Your AI Farm Loan Risk Predictor is ready!")


# =========================================================
# PREDICT RISK
# =========================================================
elif menu == "📊 Predict Risk":

    st.header("🔍 Enter Farmer Details to Predict Loan Risk")

    with st.form("loan_prediction_form"):

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # FARMER INFORMATION
        # -------------------------------------------------
        with col1:

            name = st.text_input(
                "👤 Farmer Name",
                placeholder="Enter farmer name"
            )

            age = st.number_input(
                "👴 Age",
                min_value=18,
                max_value=80,
                value=35,
                step=1
            )

            land_size = st.number_input(
                "🌾 Land Size (acres)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.1
            )

            income = st.number_input(
                "💰 Annual Income (₹)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=10000
            )

        # -------------------------------------------------
        # LOAN INFORMATION
        # -------------------------------------------------
        with col2:

            loan_amount = st.number_input(
                "🏦 Loan Amount (₹)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=10000
            )

            loan_term = st.number_input(
                "📅 Loan Term (months)",
                min_value=3,
                max_value=60,
                value=12,
                step=1
            )

            crop_type = st.selectbox(
                "🌱 Crop Type",
                [
                    "Wheat",
                    "Rice",
                    "Cotton",
                    "Sugarcane",
                    "Maize"
                ]
            )

            soil_type = st.selectbox(
                "⛏️ Soil Type",
                [
                    "Sandy",
                    "Clay",
                    "Loamy",
                    "Black",
                    "Red"
                ]
            )

            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0,
                max_value=500,
                value=120,
                step=1
            )

            previous_defaults = st.number_input(
                "⚠️ Previous Defaults",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

        submitted = st.form_submit_button(
            "🌾 Predict Risk Now",
            use_container_width=True
        )

    # =====================================================
    # RUN PREDICTION
    # =====================================================
    if submitted:

        if not name.strip():

            st.warning("⚠️ Please enter the farmer name.")

        else:

            # -------------------------------------------------
            # ENCODING
            # -------------------------------------------------
            crop_map = {
                "Wheat": 0,
                "Rice": 1,
                "Cotton": 2,
                "Sugarcane": 3,
                "Maize": 4
            }

            soil_map = {
                "Sandy": 0,
                "Clay": 1,
                "Loamy": 2,
                "Black": 3,
                "Red": 4
            }

            # Market index used by the original model
            market_index = 50

            # -------------------------------------------------
            # CREATE INPUT DATA
            # -------------------------------------------------
            input_data = pd.DataFrame(
                [[
                    age,
                    land_size,
                    income,
                    crop_map[crop_type],
                    loan_amount,
                    loan_term,
                    previous_defaults,
                    rainfall,
                    soil_map[soil_type],
                    market_index
                ]],
                columns=[
                    "age",
                    "land_size",
                    "income",
                    "crop_type",
                    "loan_amount",
                    "loan_term",
                    "previous_defaults",
                    "rainfall",
                    "soil_type",
                    "market_index"
                ]
            )

            # -------------------------------------------------
            # REORDER FEATURES TO MATCH TRAINING
            # -------------------------------------------------
            try:

                input_data = input_data[expected_features]

            except Exception as e:

                st.error(
                    "❌ The features in your scaler do not match "
                    "the features used by this application."
                )

                st.write(
                    "Features expected by scaler:"
                )

                st.write(expected_features)

                st.write(
                    "Features supplied by application:"
                )

                st.write(input_data.columns.tolist())

                st.stop()

            # -------------------------------------------------
            # SCALE INPUT
            # -------------------------------------------------
            try:

                scaled_input = scaler.transform(input_data)

            except Exception as e:

                st.error(
                    f"❌ Error while scaling the input: {e}"
                )

                st.stop()

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------
            try:

                pred_probs = model.predict_proba(
                    scaled_input
                )[0]

            except Exception as e:

                st.error(
                    f"❌ Model prediction failed: {e}"
                )

                st.stop()

            # -------------------------------------------------
            # FIND PREDICTED CLASS
            # -------------------------------------------------
            probability_index = int(
                np.argmax(pred_probs)
            )

            if hasattr(model, "classes_"):

                predicted_class = int(
                    model.classes_[probability_index]
                )

            else:

                predicted_class = probability_index

            # -------------------------------------------------
            # RISK LABEL
            # -------------------------------------------------
            if isinstance(target_mapping, dict):

                risk_label = target_mapping.get(
                    predicted_class,
                    "Unknown Risk"
                )

            else:

                try:
                    risk_label = target_mapping[
                        predicted_class
                    ]
                except Exception:
                    risk_label = "Unknown Risk"

            confidence = float(
                pred_probs[probability_index]
            )

            # -------------------------------------------------
            # SAVE CURRENT CROP
            # -------------------------------------------------
            st.session_state.current_crop = crop_type

            # -------------------------------------------------
            # SAVE HISTORY
            # -------------------------------------------------
            st.session_state.history.append(
                {
                    "Name": name,
                    "Age": age,
                    "Crop": crop_type,
                    "Loan Amount": loan_amount,
                    "Predicted Risk": risk_label,
                    "Confidence": f"{confidence:.1%}",
                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }
            )

            # =================================================
            # DISPLAY RESULT
            # =================================================
            st.markdown("---")

            result_col, gauge_col = st.columns(
                [3, 1]
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------
            with result_col:

                st.markdown(
                    f"# 🎯 {risk_label}"
                )

                st.progress(confidence)

                st.caption(
                    f"**Model Confidence:** "
                    f"{confidence:.1%}"
                )

                st.info(
                    "This confidence value represents the "
                    "model's predicted probability for the "
                    "selected class. It is not a guarantee "
                    "of repayment."
                )

            # -------------------------------------------------
            # GAUGE
            # -------------------------------------------------
            with gauge_col:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={
                            "suffix": "%"
                        },
                        title={
                            "text": "Confidence"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "bar": {
                                "color": "#1f4e79"
                            },
                            "steps": [
                                {
                                    "range": [0, 33],
                                    "color": "#ffcccc"
                                },
                                {
                                    "range": [33, 66],
                                    "color": "#fff0b3"
                                },
                                {
                                    "range": [66, 100],
                                    "color": "#ccffcc"
                                }
                            ]
                        }
                    )
                )

                gauge.update_layout(
                    height=250,
                    margin=dict(
                        l=10,
                        r=10,
                        t=40,
                        b=10
                    )
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            # =================================================
            # PROBABILITY CHART
            # =================================================
            st.markdown("---")

            st.subheader(
                "📊 Risk Probability Distribution"
            )

            if len(pred_probs) == 3:

                risk_levels = [
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ]

                colors = [
                    "#ff4444",
                    "#ffaa00",
                    "#44aa44"
                ]

            else:

                risk_levels = [
                    f"Class {i}"
                    for i in range(len(pred_probs))
                ]

                colors = [
                    "#3366cc"
                    for _ in pred_probs
                ]

            probability_chart = go.Figure(
                data=[
                    go.Bar(
                        x=risk_levels,
                        y=pred_probs,
                        marker_color=colors,
                        text=[
                            f"{p:.1%}"
                            for p in pred_probs
                        ],
                        textposition="auto"
                    )
                ]
            )

            probability_chart.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Risk Level",
                yaxis_title="Probability",
                yaxis={
                    "range": [0, 1]
                }
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True
            )

            # =================================================
            # FEATURE IMPORTANCE
            # =================================================
            if hasattr(
                model,
                "feature_importances_"
            ):

                st.markdown("---")

                st.subheader(
                    "🔍 Feature Importance"
                )

                feature_names = [
                    "Age",
                    "Land Size",
                    "Income",
                    "Crop Type",
                    "Loan Amount",
                    "Loan Term",
                    "Previous Defaults",
                    "Rainfall",
                    "Soil Type",
                    "Market Index"
                ]

                importance = (
                    model.feature_importances_
                )

                if len(feature_names) == len(
                    importance
                ):

                    importance_df = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Importance": importance
                        }
                    )

                    importance_df = (
                        importance_df
                        .sort_values(
                            "Importance",
                            ascending=True
                        )
                    )

                    feature_chart = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="viridis",
                        title=(
                            "What Drives the Model Prediction?"
                        )
                    )

                    feature_chart.update_layout(
                        height=450,
                        showlegend=False
                    )

                    st.plotly_chart(
                        feature_chart,
                        use_container_width=True
                    )


# =========================================================
# INSIGHTS
# =========================================================
elif menu == "📈 Insights":

    st.header("🌾 Agricultural Market Insights")

    @st.cache_data
    def load_dataset():

        df = pd.read_csv(
            "farm_loan_risk_dataset_with_year.csv"
        )

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        return df

    try:

        df = load_dataset()

        # -------------------------------------------------
        # CHECK DATASET
        # -------------------------------------------------
        st.subheader("📋 Dataset Overview")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Total Records",
                len(df)
            )

        with metric2:
            st.metric(
                "Total Columns",
                len(df.columns)
            )

        with metric3:

            if "crop_type" in df.columns:

                st.metric(
                    "Crop Types",
                    df["crop_type"].nunique()
                )

            else:

                st.metric(
                    "Crop Types",
                    "N/A"
                )

        st.markdown("---")

        # -------------------------------------------------
        # RISK BY CROP
        # -------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🌱 Risk Distribution by Crop"
            )

            if (
                "crop_type" in df.columns
                and "label_repaid" in df.columns
            ):

                risk_pivot = (
                    df.groupby(
                        [
                            "crop_type",
                            "label_repaid"
                        ]
                    )
                    .size()
                    .unstack(
                        fill_value=0
                    )
                )

                crop_chart = px.bar(
                    risk_pivot,
                    title="Risk Distribution by Crop",
                    height=400
                )

                crop_chart.update_layout(
                    xaxis_title="Crop Type",
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(
                    crop_chart,
                    use_container_width=True
                )

            else:

                st.warning(
                    "⚠️ crop_type or label_repaid "
                    "column is missing."
                )

        # -------------------------------------------------
        # MARKET TREND
        # -------------------------------------------------
        with col2:

            st.subheader(
                "📈 Market Trend"
            )

            if st.session_state.current_crop:

                if (
                    "crop_type" in df.columns
                    and "year" in df.columns
                    and "market_index" in df.columns
                ):

                    crop_data = df[
                        df["crop_type"]
                        == st.session_state.current_crop
                    ]

                    if len(crop_data) > 0:

                        market_data = (
                            crop_data
                            .groupby("year")
                            ["market_index"]
                            .mean()
                            .reset_index()
                        )

                        market_chart = px.line(
                            market_data,
                            x="year",
                            y="market_index",
                            markers=True,
                            height=400,
                            title=(
                                f"{st.session_state.current_crop} "
                                "Market Trend"
                            )
                        )

                        st.plotly_chart(
                            market_chart,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No market data found for "
                            "the selected crop."
                        )

                else:

                    st.warning(
                        "⚠️ Required market columns "
                        "are missing."
                    )

            else:

                st.info(
                    "👆 Make a prediction first to "
                    "view the selected crop's market trend."
                )

        # -------------------------------------------------
        # CORRELATION
        # -------------------------------------------------
        st.markdown("---")

        st.subheader(
            "📊 Feature Correlation Matrix"
        )

        numerical_columns = [
            "age",
            "land_size",
            "income",
            "loan_amount",
            "rainfall",
            "market_index"
        ]

        available_columns = [
            col
            for col in numerical_columns
            if col in df.columns
        ]

        if len(available_columns) >= 2:

            correlation = df[
                available_columns
            ].corr()

            heatmap = px.imshow(
                correlation,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix",
                height=500
            )

            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

        else:

            st.warning(
                "⚠️ Not enough numerical columns "
                "for correlation analysis."
            )

    except FileNotFoundError:

        st.error(
            "❌ Dataset file not found."
        )

        st.info(
            "Place farm_loan_risk_dataset_with_year.csv "
            "in the same folder as app.py."
        )

    except Exception as e:

        st.error(
            f"❌ Error loading dataset: {e}"
        )


# =========================================================
# VISUALS
# =========================================================
elif menu == "📊 Visuals":

    st.header("📊 Your Prediction Analytics")

    if len(st.session_state.history) == 0:

        st.info(
            "👆 Make a prediction first to unlock analytics!"
        )

    else:

        df_history = pd.DataFrame(
            st.session_state.history
        )

        # -------------------------------------------------
        # RISK DISTRIBUTION
        # -------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:

            risk_counts = (
                df_history[
                    "Predicted Risk"
                ]
                .value_counts()
            )

            pie_chart = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Distribution",
                hole=0.4,
                height=400
            )

            st.plotly_chart(
                pie_chart,
                use_container_width=True
            )

        # -------------------------------------------------
        # AGE DISTRIBUTION
        # -------------------------------------------------
        with col2:

            age_chart = px.histogram(
                df_history,
                x="Age",
                color="Predicted Risk",
                nbins=20,
                title="Age vs Risk",
                height=400
            )

            st.plotly_chart(
                age_chart,
                use_container_width=True
            )

        # -------------------------------------------------
        # LOAN AMOUNT VS AGE
        # -------------------------------------------------
        st.markdown("---")

        st.subheader(
            "💰 Loan Amount vs Age"
        )

        scatter_chart = px.scatter(
            df_history,
            x="Loan Amount",
            y="Age",
            color="Predicted Risk",
            size="Age",
            hover_name="Name",
            hover_data=["Crop"],
            title="Loan Amount vs Age vs Risk",
            height=500
        )

        st.plotly_chart(
            scatter_chart,
            use_container_width=True
        )

        # -------------------------------------------------
        # HISTORY TABLE
        # -------------------------------------------------
        st.markdown("---")

        st.subheader(
            "📋 Complete Prediction History"
        )

        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# HISTORY
# =========================================================
elif menu == "📋 History":

    st.header("🕒 Prediction History")

    if len(st.session_state.history) == 0:

        st.info(
            "📝 No predictions yet. "
            "Go to 'Predict Risk' to start!"
        )

    else:

        df_history = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------
        csv_data = (
            df_history
            .to_csv(index=False)
            .encode("utf-8")
        )

        col1, col2 = st.columns(
            [3, 1]
        )

        with col2:

            st.download_button(
                label="⬇️ Export CSV",
                data=csv_data,
                file_name="loan_risk_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        # -------------------------------------------------
        # CLEAR HISTORY
        # -------------------------------------------------
        st.markdown("---")

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.history = []

            st.success(
                "✅ Prediction history cleared."
            )

            st.rerun()
