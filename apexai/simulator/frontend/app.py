"""ApexAI Simulator - Streamlit Frontend."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Assuming this path setup is necessary for the backend engine imports
# (This path resolution depends on the environment, keeping as is)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from apexai.simulator.backend.engine import (
        DataStreamer,
        InferenceEngine,
        MultiModelTTA,
        SlidingWindowBuffer,
        TestTimeAugmentation,
    )
except ImportError:
    st.error("Failed to import backend engine. Please check your path configuration.")
    # Optionally define dummy classes to enable UI testing
    # class DataStreamer: ... (etc.)
    st.stop()


st.set_page_config(page_title="ApexAI Simulator", page_icon="🏎️", layout="wide")

st.title("🏎️ ApexAI Real-time Inference Simulator")
st.markdown("---")

# === State Reset ===
# <--- Added: Reset app_busy flag at the start of script
# This ensures buttons are always clickable after rerun
st.session_state.app_busy = False


# === Sidebar ===
st.sidebar.header("⚙️ Configuration")

# Check if config should be disabled (only when running, not just initialized)
config_disabled = st.session_state.get("running", False)

# Model mode selection
model_mode = st.sidebar.radio(
    "Model Mode",
    options=["Single Model", "Multi-Model Ensemble"],
    index=1,
    help="Single: Use one model | Ensemble: Combine predictions from two models",
    disabled=config_disabled,
)

if model_mode == "Single Model":
    config_path = st.sidebar.text_input(
        "Config Path",
        value="apexai/simulator/model_repository/prodmodel.yaml",
        help="Model path will be read from this config file",
        disabled=config_disabled,
    )
else:
    st.sidebar.subheader("Model 1 (Transformer)")
    config_path_1 = st.sidebar.text_input(
        "Config Path 1",
        value="apexai/simulator/model_repository/prodmodel1.yaml",
        help="Model path will be read from this config file",
        disabled=config_disabled,
    )

    st.sidebar.subheader("Model 2 (GRU)")
    config_path_2 = st.sidebar.text_input(
        "Config Path 2",
        value="apexai/simulator/model_repository/prodmodel2.yaml",
        help="Model path will be read from this config file",
        disabled=config_disabled,
    )

    ensemble_method = st.sidebar.selectbox(
        "Ensemble Method",
        options=["majority", "weighted"],
        index=0,
        help="majority: Simple voting | weighted: Confidence-weighted",
        disabled=config_disabled,
    )

device = st.sidebar.selectbox("Device", options=["cuda", "cpu"], index=0, disabled=config_disabled)


# === File Upload ===


# <--- Added: Callback function called when file is changed
def on_file_change():
    """Reset state and force re-initialization.

    Called when file is uploaded, changed, or deleted.
    """
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.initializing = False  # <--- Added
    st.session_state.current_file_id = None  # <--- Added (also reset ID check)
    st.session_state.inference_count = 0
    st.session_state.elapsed_time = 0.0
    # ... Reset all other states ...
    st.session_state.latest_prediction = None
    st.session_state.latest_probabilities = None
    st.session_state.tta_result = None
    st.session_state.tta_votes = None
    st.session_state.sensor_history = {
        "timestamps": [],
        "pbrake_f": [],
        "pbrake_r": [],
        "Steering_Angle": [],
        "accx_can": [],
        "accy_can": [],
        "ath": [],
        "gear": [],
        "nmot": [],
        "speed": [],
    }
    st.session_state.prediction_timeline = []
    st.session_state.tta_timeline = []
    st.session_state.per_model_predictions = None
    st.session_state.per_model_probabilities = None
    st.session_state.ensemble_metadata = None
    st.info("🔄 File has been changed. Please re-initialize the simulator.")


st.header("📁 Upload CSV Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    on_change=on_file_change,  # <--- Fixed: Register callback
    key="file_uploader",  # <--- Added: Specify key to use on_change
)

# === Session State Initialization ===
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.initializing = False  # <--- Added
    st.session_state.app_busy = False  # <--- Added
    st.session_state.inference_count = 0
    st.session_state.elapsed_time = 0.0
    st.session_state.latest_prediction = None
    st.session_state.latest_probabilities = None
    st.session_state.tta_result = None
    st.session_state.tta_votes = None
    st.session_state.sensor_history = {
        "timestamps": [],
        "pbrake_f": [],
        "pbrake_r": [],
        "Steering_Angle": [],
        "accx_can": [],
        "accy_can": [],
        "ath": [],
        "gear": [],
        "nmot": [],
        "speed": [],
    }
    st.session_state.prediction_timeline = []
    st.session_state.tta_timeline = []
    st.session_state.current_file_name = None
    st.session_state.model_mode = None
    st.session_state.per_model_predictions = None
    st.session_state.per_model_probabilities = None
    st.session_state.ensemble_metadata = None

# === File Change Detection Logic (simplified since migrated to on_change) ===
if uploaded_file is not None:
    current_file_id = id(uploaded_file)
    if st.session_state.get("current_file_id") != current_file_id:
        # on_file_change should have already reset the state,
        # but update file name and ID just in case
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.current_file_id = current_file_id
        # No need for reset logic here since it's handled by on_file_change
else:
    # If file is deleted
    if st.session_state.get("current_file_id") is not None:
        on_file_change()  # <--- Fixed: Reset on file deletion as well

# Detect model mode change and force re-initialization
if st.session_state.get("model_mode") is not None and st.session_state.model_mode != model_mode:
    st.session_state.initialized = False
    st.session_state.running = False
    st.session_state.initializing = False  # <--- Added
    st.info("🔄 Model mode changed. Please re-initialize the simulator.")


# === Initialize Button ===
# <--- Fixed: Bug 1 & 2 countermeasures
if uploaded_file is not None:
    # Conditions to disable button: (already initialized) OR (currently initializing)
    is_disabled = st.session_state.get("initialized", False) or st.session_state.get(
        "initializing", False
    )

    if st.button("🚀 Initialize Simulator", type="primary", key="init_btn", disabled=is_disabled):
        st.session_state.initializing = True  # <--- Fixed: Initialization start flag

        with st.spinner("Initializing..."):
            try:
                temp_csv_path = f"/tmp/{uploaded_file.name}"
                with open(temp_csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.model_mode = model_mode

                if model_mode == "Single Model":
                    import yaml

                    with open(config_path, encoding="utf-8") as f:  # type: ignore[assignment]
                        config = yaml.safe_load(f)
                    model_path = config["model_path"]
                    model_name = config.get("name", "Model")
                    st.session_state.engine = InferenceEngine(model_path, config_path, device)
                    st.session_state.sampling_rate = st.session_state.engine.sampling_rate
                    st.session_state.seq_len = st.session_state.engine.seq_len
                    st.session_state.feature_size = st.session_state.engine.feature_size
                    st.session_state.num_votes = st.session_state.engine.num_votes
                    feature_cols = st.session_state.engine.feature_columns
                    st.info(f"📊 Single Model Configuration ({model_name}): ...")
                else:
                    import yaml

                    with open(config_path_1, encoding="utf-8") as f:  # type: ignore[assignment]
                        config1 = yaml.safe_load(f)
                    with open(config_path_2, encoding="utf-8") as f:  # type: ignore[assignment]
                        config2 = yaml.safe_load(f)
                    model_configs = [
                        {
                            "model_path": config1["model_path"],
                            "config_path": config_path_1,
                            "name": config1.get("name", "Model1"),
                        },
                        {
                            "model_path": config2["model_path"],
                            "config_path": config_path_2,
                            "name": config2.get("name", "Model2"),
                        },
                    ]
                    st.session_state.multi_engine = MultiModelTTA(
                        model_configs,
                        device=device,
                        tta_window_size=6,
                        ensemble_method=ensemble_method,
                    )
                    first_engine = st.session_state.multi_engine.engines[0]
                    st.session_state.sampling_rate = first_engine.sampling_rate
                    st.session_state.seq_len = first_engine.seq_len
                    st.session_state.feature_size = first_engine.feature_size
                    st.session_state.num_votes = 6
                    feature_cols = first_engine.feature_columns
                    model_names = " + ".join(st.session_state.multi_engine.model_names)
                    st.info(f"📊 Multi-Model Ensemble Configuration ({model_names}): ...")

                st.session_state.streamer = DataStreamer(
                    temp_csv_path, st.session_state.sampling_rate, feature_cols
                )
                st.session_state.buffer = SlidingWindowBuffer(
                    st.session_state.seq_len, st.session_state.feature_size
                )

                if model_mode == "Single Model":
                    st.session_state.tta = TestTimeAugmentation(st.session_state.num_votes)

                st.session_state.initialized = True  # <--- Fixed: Success flag
                st.session_state.just_initialized = True

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback

                st.error(traceback.format_exc())
                st.session_state.initialized = False  # <--- Fixed: On failure

            finally:
                st.session_state.initializing = False  # <--- Fixed: Initialization end flag
                st.rerun()  # <--- Fixed: Rerun to update button state regardless of success/failure


# === Data Processing Function ===
def process_next_sample():
    """Process next sample and update session state."""
    # <--- Added: Bug 3 & 4 countermeasures
    # Don't perform any processing if app is busy (e.g., during Restart)
    if st.session_state.get("app_busy", False):
        return False

    if not st.session_state.running:
        return False

    sample = st.session_state.streamer.get_next_sample()

    if sample is None:
        st.session_state.running = False
        return False

    # ... (The rest of the processing is unchanged) ...
    st.session_state.sensor_history["timestamps"].append(st.session_state.elapsed_time)
    feature_names = [
        "pbrake_f",
        "pbrake_r",
        "Steering_Angle",
        "accx_can",
        "accy_can",
        "ath",
        "gear",
        "nmot",
        "speed",
    ]
    for i, feature_name in enumerate(feature_names):
        st.session_state.sensor_history[feature_name].append(float(sample[i]))

    for key in st.session_state.sensor_history:
        if len(st.session_state.sensor_history[key]) > 1000:
            st.session_state.sensor_history[key] = st.session_state.sensor_history[key][-1000:]

    st.session_state.buffer.add_sample(sample)

    if st.session_state.buffer.is_ready():
        sequence = st.session_state.buffer.get_sequence()

        if st.session_state.model_mode == "Single Model":
            predicted_class, probabilities = st.session_state.engine.predict(sequence)
            st.session_state.latest_prediction = predicted_class
            st.session_state.latest_probabilities = probabilities
            st.session_state.inference_count += 1
            class_name = st.session_state.engine.get_class_name(predicted_class)
            st.session_state.prediction_timeline.append(
                (st.session_state.elapsed_time, predicted_class, class_name)
            )
            st.session_state.tta.add_prediction(predicted_class)
            if st.session_state.tta.is_ready():
                majority_class, vote_counts = st.session_state.tta.get_majority_vote()
                st.session_state.tta_result = majority_class
                st.session_state.tta_votes = vote_counts
                tta_class_name = st.session_state.engine.get_class_name(majority_class)
                st.session_state.tta_timeline.append(
                    (st.session_state.elapsed_time, majority_class, tta_class_name)
                )
                st.session_state.tta.reset()
        else:
            predictions, probs, _ = st.session_state.multi_engine.predict(sequence)
            per_model_predictions = predictions
            per_model_probs = probs
            st.session_state.per_model_predictions = per_model_predictions
            st.session_state.per_model_probabilities = per_model_probs
            st.session_state.inference_count += 1
            first_engine = st.session_state.multi_engine.engines[0]
            if st.session_state.multi_engine.is_ready():
                ensemble_class, metadata = st.session_state.multi_engine.get_ensemble_result()
                st.session_state.tta_result = ensemble_class
                st.session_state.ensemble_metadata = metadata
                ensemble_class_name = first_engine.get_class_name(ensemble_class)
                st.session_state.tta_timeline.append(
                    (st.session_state.elapsed_time, ensemble_class, ensemble_class_name)
                )
                st.session_state.tta_votes = metadata.get("vote_counts", {})
        st.session_state.buffer.clear()
    st.session_state.elapsed_time += 1.0 / st.session_state.sampling_rate
    return True


# === Initialization Complete Message ===
if st.session_state.get("initialized", False) and st.session_state.get("just_initialized", False):
    st.success("✅ Initialization completed successfully!")
    st.session_state.just_initialized = False

# === Simulation Control ===
if st.session_state.get("initialized", False):
    st.markdown("---")
    st.header("▶️ Simulation Control")

    col1, col2, col3 = st.columns([1, 1, 1])

    # <--- Fixed: Bug 3 & 4 countermeasures (added app_busy flag)
    with col1:
        if st.button(
            "▶️ Start",
            disabled=st.session_state.running or st.session_state.app_busy,
            key="start_btn",
        ):
            st.session_state.app_busy = True  # <--- Added
            st.session_state.running = True
            st.rerun()

    with col2:
        if st.button(
            "⏸️ Pause",
            disabled=not st.session_state.running or st.session_state.app_busy,
            key="pause_btn",
        ):
            st.session_state.app_busy = True  # <--- Added
            st.session_state.running = False
            st.rerun()

    with col3:
        if st.button("🔄 Restart", key="restart_btn", disabled=st.session_state.app_busy):
            st.session_state.app_busy = True  # <--- Added
            st.session_state.running = False

            # Reset state
            st.session_state.inference_count = 0
            st.session_state.elapsed_time = 0.0
            st.session_state.latest_prediction = None
            st.session_state.latest_probabilities = None
            st.session_state.tta_result = None
            st.session_state.tta_votes = None
            st.session_state.per_model_predictions = None
            st.session_state.per_model_probabilities = None
            st.session_state.ensemble_metadata = None
            for key in st.session_state.sensor_history:
                st.session_state.sensor_history[key] = []
            st.session_state.prediction_timeline = []
            st.session_state.tta_timeline = []

            # Reset engine/buffer
            st.session_state.streamer.reset()
            st.session_state.buffer.clear()
            if st.session_state.model_mode == "Single Model":
                st.session_state.tta.reset()
            else:
                st.session_state.multi_engine.reset()

            st.rerun()

    # Fragment 1: Real-time Results
    st.markdown("---")
    st.header("📊 Real-time Results")

    @st.fragment(run_every=0.1 if st.session_state.running else None)
    def show_realtime_results():
        """Display real-time inference results - 10Hz update."""
        if st.session_state.running:
            process_next_sample()  # <--- Fixed: process_next_sample checks running state internally

        # ... (The rest of the display logic is unchanged) ...
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("⏱️ Time", f"{st.session_state.elapsed_time:.1f}s")
        with metric_col2:
            st.metric("🔢 Inferences", st.session_state.inference_count)
        with metric_col3:
            class_name = "N/A"
            if st.session_state.tta_result is not None:
                engine = (
                    st.session_state.engine
                    if st.session_state.model_mode == "Single Model"
                    else st.session_state.multi_engine.engines[0]
                )
                class_name = engine.get_class_name(st.session_state.tta_result)
            st.metric("🎯 Final Prediction", class_name)

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.subheader("🤖 Latest Inference")
            if st.session_state.model_mode == "Single Model":
                if st.session_state.latest_probabilities is not None:
                    probs = st.session_state.latest_probabilities
                    top5_indices = probs.argsort()[-5:][::-1]
                    st.write("**Top 5 Predictions:**")
                    for idx in top5_indices:
                        class_name = st.session_state.engine.get_class_name(int(idx))
                        st.write(f"{class_name}: {probs[idx]:.4f}")
                else:
                    st.info("Waiting for first inference...")
            else:
                if st.session_state.per_model_predictions is not None:
                    for i, model_name in enumerate(st.session_state.multi_engine.model_names):
                        st.write(f"**{model_name}:**")
                        probs = st.session_state.per_model_probabilities[i]
                        pred_class = st.session_state.per_model_predictions[i]
                        top3_indices = probs.argsort()[-3:][::-1]
                        first_engine = st.session_state.multi_engine.engines[0]
                        for idx in top3_indices:
                            class_name = first_engine.get_class_name(int(idx))
                            marker = "→" if int(idx) == pred_class else " "
                            st.write(f"{marker} {class_name}: {probs[idx]:.4f}")
                        st.write("")
                else:
                    st.info("Waiting for first inference...")

        with result_col2:
            st.subheader("🎯 Final Prediction Details")
            if st.session_state.tta_votes is not None:
                st.write("**Vote Counts:**")
                engine = (
                    st.session_state.engine
                    if st.session_state.model_mode == "Single Model"
                    else st.session_state.multi_engine.engines[0]
                )
                for class_id, count in sorted(st.session_state.tta_votes.items()):
                    class_name = engine.get_class_name(class_id)
                    st.write(f"{class_name}: {count} votes")

                if (
                    st.session_state.model_mode != "Single Model"
                    and st.session_state.ensemble_metadata is not None
                ):
                    metadata = st.session_state.ensemble_metadata
                    st.write("")
                    agreement_rate = metadata.get("agreement_rate", 0.0)
                    st.write(f"**Model Agreement:** {agreement_rate * 100:.1f}%")

                    # <--- Fixed: Use .get() for safe access
                    total_votes = metadata.get("total_votes", "N/A")  # Default value to 'N/A'
                    st.write(f"**Total Votes:** {total_votes}")
            else:
                st.info("Waiting for TTA results...")

    show_realtime_results()

    # Fragment 2: Sensor data graph
    st.markdown("---")
    st.header("📈 Sensor Data Visualization")

    @st.fragment(run_every=1.0 if st.session_state.running else None)
    def show_live_chart():
        """Display real-time graph - 1Hz update."""
        if len(st.session_state.sensor_history["timestamps"]) > 0:
            max_samples = 500
            timestamps_all = st.session_state.sensor_history["timestamps"]

            if len(timestamps_all) > max_samples:
                timestamps = timestamps_all[-max_samples:]
                steering = st.session_state.sensor_history["Steering_Angle"][-max_samples:]
                throttle = st.session_state.sensor_history["ath"][-max_samples:]
                brake_f = st.session_state.sensor_history["pbrake_f"][-max_samples:]
                brake_r = st.session_state.sensor_history["pbrake_r"][-max_samples:]
                min_time = timestamps[0]
            else:
                timestamps = timestamps_all
                steering = st.session_state.sensor_history["Steering_Angle"]
                throttle = st.session_state.sensor_history["ath"]
                brake_f = st.session_state.sensor_history["pbrake_f"]
                brake_r = st.session_state.sensor_history["pbrake_r"]
                min_time = timestamps[0] if timestamps else 0

            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    "Steering Angle (deg)",
                    "Throttle (%)",
                    "Brake Pressure (front/rear)",
                ),
                shared_xaxes=True,
                vertical_spacing=0.1,
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=steering,
                    mode="lines",
                    name="Steering Angle",
                    line=dict(color="blue", width=2),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=throttle,
                    mode="lines",
                    name="Throttle",
                    line=dict(color="green", width=2),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=brake_f,
                    mode="lines",
                    name="Brake Front",
                    line=dict(color="red", width=2),
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=brake_r,
                    mode="lines",
                    name="Brake Rear",
                    line=dict(color="orange", width=2),
                ),
                row=3,
                col=1,
            )

            pred_times = [p[0] for p in st.session_state.prediction_timeline if p[0] >= min_time]
            for pred_time in pred_times:
                for r in range(1, 4):
                    fig.add_vline(
                        x=pred_time,
                        line_width=1,
                        line_dash="dash",
                        line_color="lightgray",
                        opacity=0.5,
                        row=r,
                        col=1,
                    )

            tta_times = [t[0] for t in st.session_state.tta_timeline if t[0] >= min_time]
            for tta_time in tta_times:
                for r in range(1, 4):
                    fig.add_vline(
                        x=tta_time,
                        line_width=2,
                        line_dash="solid",
                        line_color="red",
                        opacity=0.7,
                        row=r,
                        col=1,
                    )

            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="Angle (deg)", row=1, col=1)
            fig.update_yaxes(title_text="Throttle (%)", row=2, col=1)
            fig.update_yaxes(title_text="Pressure (psi)", row=3, col=1)
            fig.update_layout(
                height=800,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Collecting data...")

    show_live_chart()

# === Before Initialization ===
else:
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file")
    else:
        # This message is covered by on_file_change and Initialize button logic
        # st.info("👆 Click Initialize Simulator")
        pass
