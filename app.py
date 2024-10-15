import numpy as np
import pandas as pd
from river import time_series
from river import anomaly
from river import preprocessing
from river import linear_model
from river import optim
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Output, Input, State
import threading
import queue
import time
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Data Stream Simulator Class
# -----------------------------


class DataStreamSimulator:
    """
    Simulates a real-time data stream with concept drift, seasonal patterns, and anomalies.
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.current_mean = 0
        self.current_std = 1
        self.change_interval = 100
        self.anomaly_interval = 50
        self.step = 0
        self.streaming = False
        self.lock = threading.Lock()
        self.seasonality_period = 200

    def simulate_stream(self):
        """
        Generator that yields data points with dynamic changes, seasonal patterns, and anomalies.
        """
        while True:
            with self.lock:
                if not self.streaming:
                    time.sleep(0.1)
                    continue
                current_step = self.step

            # Introduce seasonality
            seasonal = np.sin(2 * np.pi * current_step / self.seasonality_period)

            # Change mean and std at intervals to simulate concept drift
            if current_step % self.change_interval == 0 and current_step != 0:
                with self.lock:
                    self.current_mean += np.random.uniform(-1.0, 1.0)
                    self.current_std = np.random.uniform(0.5, 1.5)

            # Generate data point with seasonality and noise
            data_point = np.random.normal(
                self.current_mean + seasonal, self.current_std
            )

            # Introduce anomalies at specified intervals
            if current_step % self.anomaly_interval == 0 and current_step != 0:
                anomaly = np.random.uniform(-10, 10)
                data_point += anomaly

            with self.lock:
                self.step += 1

            yield data_point
            time.sleep(0.05)


# -----------------------------
# Anomaly Detector Class
# -----------------------------


class StreamingAnomalyDetector:
    """
    Detects anomalies in streaming data using River's Predictive Anomaly Detection.
    """

    def __init__(self):
        # Initialize the predictive model for time series
        self.predictive_model = time_series.SNARIMAX(
            p=12,
            d=1,
            q=12,
            m=12,
            sd=1,
            regressor=(
                preprocessing.StandardScaler()
                | linear_model.LinearRegression(
                    optimizer=optim.SGD(0.005),
                )
            ),
        )
        # Initialize Predictive Anomaly Detection
        self.model = anomaly.PredictiveAnomalyDetection(
            predictive_model=self.predictive_model,
            horizon=1,
            n_std=3.5,
            warmup_period=15,
        )
        self.anomalies = []

        # Define a threshold for anomaly detection
        self.threshold = 0.5

    def update(self, data_point):
        """
        Updates the model with the new data point and detects anomalies.
        """
        # Calculate the anomaly score
        score = self.model.score_one(None, data_point)

        # Update the model with the new data point
        self.model.learn_one(None, data_point)

        # Determine if the current data point is an anomaly based on the threshold
        is_anomaly = score > self.threshold

        if is_anomaly:
            self.anomalies.append({"time": None, "value": data_point, "score": score})

        return is_anomaly, score


# -----------------------------
# Global Variables for Data Storage
# -----------------------------

# Initialize DataFrame globally
df = pd.DataFrame(columns=["time", "value", "is_anomaly", "anomaly_score"])

# Initialize simulator, detector, data queue, and stop event globally
simulator = DataStreamSimulator()
detector = StreamingAnomalyDetector()
data_queue = queue.Queue()
stop_event = threading.Event()

# -----------------------------
# Data Streaming and Processing
# -----------------------------


def data_streaming(simulator, detector, data_queue, stop_event):
    """
    Streams data from the simulator and processes it using the detector.
    """
    try:
        stream = simulator.simulate_stream()
        while not stop_event.is_set():
            data_point = next(stream)
            is_anomaly, anomaly_score = detector.update(data_point)
            data_queue.put(
                {
                    "time": simulator.step,
                    "value": data_point,
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                }
            )
    except Exception as e:
        print(f"Error in data_streaming: {e}")


# -----------------------------
# Visualization with Dash
# -----------------------------

app = Dash(__name__)

# Button styles
button_style = {
    "padding": "10px",
    "border": "none",
    "border-radius": "12px",
    "font-size": "16px",
    "margin-right": "10px",
    "cursor": "pointer",
    "transition": "transform 0.3s",
}

# Define the layout outside of any function for proper callback access
app.layout = html.Div(
    style={
        "background-color": "#f9f9f9",
        "padding": "20px",
        "border-radius": "20px",
        "max-width": "1000px",
        "margin": "auto",
        "font-family": "Arial, sans-serif",
        "color": "#333333",
        "box-shadow": "0 4px 30px rgba(0, 0, 0, 0.1)",
    },
    children=[
        html.H1(
            "Real-Time Anomaly Detection",
            style={
                "text-align": "center",
                "color": "#2c3e50",
                "font-family": "Calibri",
            },
        ),
        dcc.Graph(id="live-graph"),
        dcc.Interval(id="graph-update", interval=500, n_intervals=0),
        html.Div(
            [
                dcc.Checklist(
                    id="toggle-anomalies",
                    options=[
                        {"label": "Show Anomaly Regions", "value": "anomalies"},
                        {"label": "Show Trend Line", "value": "trend_line"},
                    ],
                    value=[],
                    inline=True,
                ),
                html.Button(
                    "Start Streaming",
                    id="start-btn",
                    n_clicks=0,
                    style={
                        **button_style,
                        "background-color": "#4CAF50",
                        "color": "white",
                    },
                ),
                html.Button(
                    "Stop Streaming",
                    id="stop-btn",
                    n_clicks=0,
                    style={
                        **button_style,
                        "background-color": "#f44336",
                        "color": "white",
                        "display": "none",
                    },
                ),
                html.Button(
                    "Reset",
                    id="reset-btn",
                    n_clicks=0,
                    style={
                        **button_style,
                        "background-color": "#FF9800",
                        "color": "white",
                        "display": "none",
                    },
                ),
                html.Button(
                    "Download Anomalies CSV",
                    id="download-btn",
                    n_clicks=0,
                    style={
                        **button_style,
                        "background-color": "#9C27B0",
                        "color": "white",
                        "display": "none",
                    },
                ),
            ],
            style={"text-align": "center", "margin-top": "20px"},
        ),
        dcc.Download(id="download-dataframe-csv"),
        html.Div(id="status", style={"text-align": "center", "margin-top": "10px"}),
        # Hidden Div for managing streaming
        html.Div(id="reset-data", style={"display": "none"}),
        # Store to keep the thread reference
        dcc.Store(id="thread-store", data={"streaming_thread": None}),
    ],
)


# Callback to update the live graph
@app.callback(
    Output("live-graph", "figure"),
    [Input("graph-update", "n_intervals"), Input("toggle-anomalies", "value")],
)
def update_graph(n, toggle_values):
    global df
    try:
        while not data_queue.empty():
            data = data_queue.get()
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        if df.empty:
            return go.Figure()

        df_current = df.tail(1000)

        fig = make_subplots(rows=1, cols=1)

        # Add the main data stream
        fig.add_trace(
            go.Scatter(
                x=df_current["time"],
                y=df_current["value"],
                mode="lines",
                name="Data Stream",
            )
        )

        # Highlight anomalies if the checkbox is checked
        if "anomalies" in toggle_values:
            anomalies = df_current[df_current["is_anomaly"] == True]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["time"],
                        y=anomalies["value"],
                        mode="markers",
                        marker=dict(color="red", size=8, symbol="circle-open"),
                        name="Anomalies",
                    )
                )
                # Highlight the region of anomalies
                for _, anomaly in anomalies.iterrows():
                    fig.add_shape(
                        type="rect",
                        x0=anomaly["time"] - 0.5,
                        x1=anomaly["time"] + 0.5,
                        y0=df_current["value"].min() - 1,
                        y1=df_current["value"].max() + 1,
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        line=dict(color="red", width=0),
                    )

        # Add a dashed trend line if the checkbox is checked
        if "trend_line" in toggle_values:
            df_current["trend"] = (
                df_current["value"].rolling(window=10).mean()
            )  # Simple moving average
            fig.add_trace(
                go.Scatter(
                    x=df_current["time"],
                    y=df_current["trend"],
                    mode="lines",
                    name="Trend Line",
                    line=dict(dash="dash", color="green"),
                )
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            xaxis=dict(range=[df_current["time"].min(), df_current["time"].max()]),
            yaxis=dict(
                range=[df_current["value"].min() - 1, df_current["value"].max() + 1]
            ),
            showlegend=True,
            template="plotly_white",
        )

        return fig
    except Exception as e:
        print(f"Error in update_graph: {e}")
        return go.Figure()


# Callback to toggle button visibility
@app.callback(
    [
        Output("download-btn", "style"),
        Output("reset-btn", "style"),
        Output("stop-btn", "style"),
        Output("start-btn", "style"),
    ],
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def toggle_buttons(start_n, stop_n, reset_n):
    ctx = callback_context
    if not ctx.triggered:
        return (
            {"display": "none"},
            {"display": "none"},
            {"display": "none"},
            {
                "display": "inline-block",
                **button_style,
                "background-color": "#4CAF50",
                "color": "white",
            },
        )

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "start-btn":
        return (
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#9C27B0",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#f44336",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
        )
    elif button_id == "stop-btn":
        return (
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#9C27B0",
                "color": "white",
            },
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#FF9800",
                "color": "white",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "none",
            },
        )
    elif button_id == "reset-btn":
        return (
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "none",
            },
            {
                **button_style,
                "display": "inline-block",
                "background-color": "#4CAF50",
                "color": "white",
            },
        )

    return (
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
        {
            "display": "inline-block",
            **button_style,
            "background-color": "#4CAF50",
            "color": "white",
        },
    )


# Callback to handle CSV download
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks")],
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    """
    Handles the download of the anomalies DataFrame as a CSV file.
    """
    try:
        if df.empty:
            return dash.no_update
        anomalies_df = df[df["is_anomaly"] == True]
        if anomalies_df.empty:
            return dash.no_update
        return dcc.send_data_frame(anomalies_df.to_csv, "anomalies.csv", index=False)
    except Exception as e:
        print(f"Error in download_csv: {e}")
        return dash.no_update


# Callback to control button disabling
@app.callback(
    [
        Output("start-btn", "disabled"),
        Output("stop-btn", "disabled"),
        Output("reset-btn", "disabled"),
        Output("download-btn", "disabled"),
    ],
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def control_buttons(start_n, stop_n, reset_n):
    """
    Controls the enabling and disabling of buttons based on user interactions.
    """
    ctx = callback_context

    if not ctx.triggered:
        return False, True, True, True

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "start-btn":
        return True, False, True, True
    elif button_id == "stop-btn":
        return False, True, False, False
    elif button_id == "reset-btn":
        return False, True, True, True
    return False, True, True, True


# Callback to update the start button label
@app.callback(
    Output("start-btn", "children"),
    [Input("start-btn", "n_clicks"), Input("stop-btn", "n_clicks")],
)
def update_start_button_label(start_clicks, stop_clicks):
    """
    Updates the label of the start button based on its state.
    """
    ctx = callback_context
    if not ctx.triggered:
        return "Start Streaming"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return "Streaming..."
    elif button_id == "stop-btn":
        return "Start Streaming"
    return "Start Streaming"


# Callback to update the status text
@app.callback(
    Output("status", "children"),
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_status(start_n, stop_n, reset_n):
    """
    Updates the status text displayed to the user based on button interactions.
    """
    ctx = callback_context
    if not ctx.triggered:
        return "Status: Stopped"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-btn":
        return "Status: Streaming..."
    elif button_id == "stop-btn":
        return "Status: Stopped"
    elif button_id == "reset-btn":
        return "Status: Reset"
    return "Status: Stopped"


# Callback to manage streaming actions
@app.callback(
    Output("reset-data", "children"),
    [
        Input("start-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
    ],
    [
        State("thread-store", "data"),
    ],
    prevent_initial_call=True,
)
def manage_streaming(start_n, stop_n, reset_n, store_data):
    """
    Manages the streaming actions based on button clicks.
    """
    global df
    ctx = callback_context

    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "start-btn":
        simulator.streaming = True
        stop_event.clear()
        streaming_thread = threading.Thread(
            target=data_streaming,
            args=(simulator, detector, data_queue, stop_event),
            daemon=True,
        )
        streaming_thread.start()
        store_data["streaming_thread"] = streaming_thread
    elif button_id == "stop-btn":
        simulator.streaming = False
        stop_event.set()
        streaming_thread = store_data.get("streaming_thread")
        if streaming_thread is not None and streaming_thread.is_alive():
            streaming_thread.join()
    elif button_id == "reset-btn":
        simulator.streaming = False
        stop_event.set()
        streaming_thread = store_data.get("streaming_thread")
        if streaming_thread is not None and streaming_thread.is_alive():
            streaming_thread.join()
        df = pd.DataFrame(columns=["time", "value", "is_anomaly", "anomaly_score"])
        detector.anomalies.clear()

    return ""


# -----------------------------
# Main Execution Function
# -----------------------------


def main():
    """
    Main function to start data streaming and visualization.
    """
    app.run_server(debug=False)


if __name__ == "__main__":
    main()
