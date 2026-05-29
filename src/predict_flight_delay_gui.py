from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk, messagebox


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*"
)


class TabTransformerModel(nn.Module):
    def __init__(
        self,
        categorical_cardinalities,
        num_numeric_features,
        num_classes,
        d_model=96,
        n_heads=6,
        n_layers=4,
        ff_dim=384,
        transformer_dropout=0.12,
        head_hidden_dims=[384, 192],
        head_dropout=0.20
    ):
        super().__init__()

        self.num_categorical_features = len(categorical_cardinalities)
        self.num_numeric_features = num_numeric_features
        self.d_model = d_model

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model)
            for cardinality in categorical_cardinalities
        ])

        self.cat_column_embeddings = nn.Parameter(
            torch.randn(self.num_categorical_features, d_model) * 0.02
        )

        self.num_column_embeddings = nn.Parameter(
            torch.randn(num_numeric_features, d_model) * 0.02
        )

        self.num_weight = nn.Parameter(
            torch.randn(num_numeric_features, d_model) * 0.02
        )

        self.num_bias = nn.Parameter(
            torch.zeros(num_numeric_features, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        head_input_dim = (2 * d_model) + num_numeric_features

        head_layers = []
        input_dim = head_input_dim

        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.LayerNorm(input_dim))
            head_layers.append(nn.Linear(input_dim, hidden_dim))
            head_layers.append(nn.GELU())
            head_layers.append(nn.Dropout(head_dropout))
            input_dim = hidden_dim

        head_layers.append(nn.LayerNorm(input_dim))
        head_layers.append(nn.Linear(input_dim, num_classes))

        self.head = nn.Sequential(*head_layers)

    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)

        cat_tokens = []

        for i, embedding_layer in enumerate(self.cat_embeddings):
            token = embedding_layer(x_cat[:, i])
            token = token + self.cat_column_embeddings[i]
            cat_tokens.append(token.unsqueeze(1))

        cat_tokens = torch.cat(cat_tokens, dim=1)

        num_tokens = (
            x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)
            + self.num_bias.unsqueeze(0)
            + self.num_column_embeddings.unsqueeze(0)
        )

        cls_token = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat(
            [cls_token, cat_tokens, num_tokens],
            dim=1
        )

        tokens = self.input_norm(tokens)

        encoded_tokens = self.transformer(tokens)

        cls_output = encoded_tokens[:, 0, :]
        mean_output = encoded_tokens[:, 1:, :].mean(dim=1)

        head_input = torch.cat(
            [cls_output, mean_output, x_num],
            dim=1
        )

        logits = self.head(head_input)

        return logits


def hhmm_to_minutes(time_text):
    hour, minute = time_text.split(":")
    return int(hour) * 60 + int(minute)


def minutes_to_time_block(minutes):
    hour = minutes // 60
    return f"{hour:02d}00-{hour:02d}59"


def build_time_options(step_minutes=15):
    options = []

    for minutes in range(0, 24 * 60, step_minutes):
        hour = minutes // 60
        minute = minutes % 60
        options.append(f"{hour:02d}:{minute:02d}")

    return options


def build_features(input_data):
    dep_minutes = hhmm_to_minutes(input_data["departure_time"])
    arr_minutes = hhmm_to_minutes(input_data["arrival_time"])

    month = int(input_data["month"])
    day_of_week = int(input_data["day_of_week"])

    origin = input_data["origin"]
    dest = input_data["dest"]

    features = {}

    features["OP_UNIQUE_CARRIER"] = input_data["carrier"]

    features["ORIGIN"] = origin
    features["ORIGIN_STATE_ABR"] = input_data["origin_state"]

    features["DEST"] = dest
    features["DEST_STATE_ABR"] = input_data["dest_state"]

    features["ROUTE"] = f"{origin}_{dest}"

    features["DEP_TIME_BLK"] = minutes_to_time_block(dep_minutes)
    features["ARR_TIME_BLK"] = minutes_to_time_block(arr_minutes)

    features["DISTANCE"] = float(input_data["distance"])

    features["CRS_DEP_MINUTES_OF_DAY"] = dep_minutes

    features["dep_time_sin"] = np.sin(2 * np.pi * dep_minutes / 1440)
    features["dep_time_cos"] = np.cos(2 * np.pi * dep_minutes / 1440)

    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    features["is_weekend"] = 1 if day_of_week in [6, 7] else 0

    return features


class FlightDelayPredictor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

        self.processed_dir = self.base_dir / "data" / "generated" / "processed"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metadata = self.load_json(self.processed_dir / "metadata.json")
        self.category_cardinalities = self.load_json(
            self.processed_dir / "category_cardinalities.json"
        )
        self.tabtransformer_results = self.load_json(
            self.results_dir / "tabtransformer_results.json"
        )

        with open(self.processed_dir / "category_encoders.pkl", "rb") as f:
            self.category_encoders = pickle.load(f)

        with open(self.processed_dir / "numeric_scaler.pkl", "rb") as f:
            self.numeric_scaler = pickle.load(f)

        self.checkpoint = torch.load(
            self.models_dir / "tabtransformer_best.pt",
            map_location=self.device
        )

        self.model = self.load_model()
        self.class_names = self.metadata["class_names"]
        self.categorical_cols = self.metadata["categorical_cols"]
        self.numeric_cols = self.metadata["numeric_cols"]

        self.best_multipliers = self.tabtransformer_results.get(
            "best_multipliers",
            [1.0, 1.0, 1.0]
        )

    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def load_model(self):
        model = TabTransformerModel(
            categorical_cardinalities=self.checkpoint["categorical_cardinalities"],
            num_numeric_features=len(self.checkpoint["numeric_cols"]),
            num_classes=self.checkpoint["num_classes"],
            **self.checkpoint["model_config"]
        )

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def prepare_input(self, features):
        x_cat = []

        for col in self.categorical_cols:
            value = str(features[col])
            encoder = self.category_encoders[col]
            encoded_value = encoder.get(value, 0)
            x_cat.append(encoded_value)

        x_cat = np.array([x_cat], dtype=np.int64)

        x_num_df = pd.DataFrame(
            [
                {
                    col: float(features[col])
                    for col in self.numeric_cols
                }
            ],
            columns=self.numeric_cols
        )

        x_num = self.numeric_scaler.transform(x_num_df).astype(np.float32)

        return x_cat, x_num

    def predict(self, input_data):
        features = build_features(input_data)

        x_cat, x_num = self.prepare_input(features)

        x_cat_tensor = torch.tensor(
            x_cat,
            dtype=torch.long
        ).to(self.device)

        x_num_tensor = torch.tensor(
            x_num,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(x_cat_tensor, x_num_tensor)
            raw_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        adjusted_scores = raw_probs * np.array(self.best_multipliers)
        calibrated_probs = adjusted_scores / adjusted_scores.sum()

        predicted_class = int(np.argmax(calibrated_probs))

        return {
            "predicted_class": predicted_class,
            "predicted_label": self.class_names[str(predicted_class)],
            "calibrated_probabilities": {
                self.class_names[str(i)]: float(calibrated_probs[i])
                for i in range(len(calibrated_probs))
            },
            "features_used": features
        }

    def get_summary(self):
        summary = {
            "device": str(self.device),
            "categorical_features": len(self.categorical_cols),
            "numerical_features": len(self.numeric_cols),
            "num_classes": int(self.metadata["num_classes"]),
            "model_name": self.tabtransformer_results.get(
                "model_name",
                "TabTransformer"
            )
        }

        train_path = self.processed_dir / "train_data.npz"
        val_path = self.processed_dir / "val_data.npz"
        test_path = self.processed_dir / "test_data.npz"

        if train_path.exists() and val_path.exists() and test_path.exists():
            train_data = np.load(train_path)
            val_data = np.load(val_path)
            test_data = np.load(test_path)

            train_rows = int(train_data["y"].shape[0])
            val_rows = int(val_data["y"].shape[0])
            test_rows = int(test_data["y"].shape[0])

            summary["train_rows"] = train_rows
            summary["validation_rows"] = val_rows
            summary["test_rows"] = test_rows
            summary["total_rows"] = train_rows + val_rows + test_rows

        calibrated_metrics = self.tabtransformer_results.get(
            "calibrated_test_metrics",
            {}
        )

        raw_metrics = self.tabtransformer_results.get(
            "raw_test_metrics",
            {}
        )

        summary["calibrated_accuracy"] = calibrated_metrics.get("accuracy")
        summary["calibrated_macro_f1"] = calibrated_metrics.get("macro_f1")
        summary["calibrated_weighted_f1"] = calibrated_metrics.get("weighted_f1")
        summary["raw_macro_f1"] = raw_metrics.get("macro_f1")

        return summary


class FlightDelayApp:
    def __init__(self, root, predictor, route_lookup):
        self.root = root
        self.predictor = predictor
        self.route_lookup = route_lookup.copy()

        self.root.title("Flight Delay Prediction - Calibrated TabTransformer")
        self.root.geometry("1240x720")
        self.root.minsize(1180, 680)

        self.setup_theme()

        self.month_options = [str(i) for i in range(1, 13)]

        self.day_options = [
            "1 - Monday",
            "2 - Tuesday",
            "3 - Wednesday",
            "4 - Thursday",
            "5 - Friday",
            "6 - Saturday",
            "7 - Sunday"
        ]

        self.time_options = build_time_options(step_minutes=15)

        self.carrier_options = sorted(
            list(self.predictor.category_encoders["OP_UNIQUE_CARRIER"].keys())
        )

        self.route_lookup["display_route"] = (
            self.route_lookup["ORIGIN"].astype(str)
            + " -> "
            + self.route_lookup["DEST"].astype(str)
            + " | "
            + self.route_lookup["ORIGIN_CITY_NAME"].astype(str)
            + " to "
            + self.route_lookup["DEST_CITY_NAME"].astype(str)
        )

        self.route_options = self.route_lookup["display_route"].tolist()

        self.create_widgets()

    def setup_theme(self):
        self.colors = {
            "bg": "#EEF3FA",
            "surface": "#FFFFFF",
            "surface_alt": "#F8FAFC",
            "primary": "#1E40AF",
            "primary_light": "#DBEAFE",
            "primary_dark": "#1E3A8A",
            "accent": "#2563EB",
            "text": "#0F172A",
            "muted": "#64748B",
            "border": "#CBD5E1",
            "success": "#16A34A",
            "warning": "#D97706",
            "danger": "#DC2626",
            "light_success": "#DCFCE7",
            "light_warning": "#FEF3C7",
            "light_danger": "#FEE2E2",
            "neutral": "#E2E8F0"
        }

        self.root.configure(bg=self.colors["bg"])

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground=self.colors["text"],
            padding=7,
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"]
        )

        style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground="white",
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 11, "bold"),
            padding=(18, 10)
        )

        style.map(
            "Primary.TButton",
            background=[("active", self.colors["primary_dark"])],
            foreground=[("active", "white")]
        )

        style.configure(
            "Green.Horizontal.TProgressbar",
            troughcolor="#E5E7EB",
            background=self.colors["success"],
            bordercolor="#E5E7EB",
            lightcolor=self.colors["success"],
            darkcolor=self.colors["success"]
        )

        style.configure(
            "Orange.Horizontal.TProgressbar",
            troughcolor="#E5E7EB",
            background=self.colors["warning"],
            bordercolor="#E5E7EB",
            lightcolor=self.colors["warning"],
            darkcolor=self.colors["warning"]
        )

        style.configure(
            "Red.Horizontal.TProgressbar",
            troughcolor="#E5E7EB",
            background=self.colors["danger"],
            bordercolor="#E5E7EB",
            lightcolor=self.colors["danger"],
            darkcolor=self.colors["danger"]
        )

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill="both", expand=True, padx=22, pady=18)

        self.create_header(main_container)

        body = tk.Frame(main_container, bg=self.colors["bg"])
        body.pack(fill="both", expand=True)

        left_panel = tk.Frame(body, bg=self.colors["bg"], width=500)
        left_panel.pack(side="left", fill="y", padx=(0, 14))
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(body, bg=self.colors["bg"])
        right_panel.pack(side="left", fill="both", expand=True)

        self.create_input_card(left_panel)
        self.create_route_card(left_panel)
        self.create_button_card(left_panel)

        self.create_prediction_card(right_panel)
        self.create_project_summary_card(right_panel)

        self.set_defaults()

    def create_header(self, parent):
        header = tk.Frame(parent, bg=self.colors["primary"], height=86)
        header.pack(fill="x", pady=(0, 16))
        header.pack_propagate(False)

        left = tk.Frame(header, bg=self.colors["primary"])
        left.pack(side="left", fill="both", expand=True, padx=22)

        tk.Label(
            left,
            text="Flight Delay Prediction",
            font=("Segoe UI", 22, "bold"),
            bg=self.colors["primary"],
            fg="white"
        ).pack(anchor="w", pady=(13, 0))

        tk.Label(
            left,
            text="Calibrated TabTransformer model for multiclass flight delay severity prediction",
            font=("Segoe UI", 10),
            bg=self.colors["primary"],
            fg="#DBEAFE"
        ).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(header, bg=self.colors["primary"])
        right.pack(side="right", padx=22)

        tk.Label(
            right,
            text="MODEL READY",
            font=("Segoe UI", 10, "bold"),
            bg="#DBEAFE",
            fg=self.colors["primary_dark"],
            padx=12,
            pady=6
        ).pack(pady=24)

    def create_card(self, parent, title):
        card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        card.pack(fill="x", pady=(0, 14))

        tk.Label(
            card,
            text=title,
            font=("Segoe UI", 14, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"]
        ).pack(anchor="w", padx=18, pady=(16, 8))

        return card

    def create_input_card(self, parent):
        input_card = self.create_card(parent, "Flight Information")

        form_frame = tk.Frame(input_card, bg=self.colors["surface"])
        form_frame.pack(fill="x", padx=18, pady=(0, 16))

        self.carrier_var = tk.StringVar()
        self.route_var = tk.StringVar()
        self.month_var = tk.StringVar()
        self.day_var = tk.StringVar()
        self.dep_time_var = tk.StringVar()
        self.arr_time_var = tk.StringVar()

        self.add_combobox(
            form_frame,
            label="Carrier",
            variable=self.carrier_var,
            values=self.carrier_options,
            row=0,
            width=34
        )

        self.add_combobox(
            form_frame,
            label="Route",
            variable=self.route_var,
            values=self.route_options,
            row=1,
            width=48
        )

        self.add_combobox(
            form_frame,
            label="Month",
            variable=self.month_var,
            values=self.month_options,
            row=2,
            width=18
        )

        self.add_combobox(
            form_frame,
            label="Day of week",
            variable=self.day_var,
            values=self.day_options,
            row=3,
            width=28
        )

        self.add_combobox(
            form_frame,
            label="Departure time",
            variable=self.dep_time_var,
            values=self.time_options,
            row=4,
            width=18
        )

        self.add_combobox(
            form_frame,
            label="Arrival time block",
            variable=self.arr_time_var,
            values=self.time_options,
            row=5,
            width=18
        )

    def create_route_card(self, parent):
        route_card = self.create_card(parent, "Selected Route")

        self.origin_info_var = tk.StringVar(value="Origin: -")
        self.dest_info_var = tk.StringVar(value="Destination: -")
        self.distance_info_var = tk.StringVar(value="Distance: -")

        info_frame = tk.Frame(route_card, bg=self.colors["surface"])
        info_frame.pack(fill="x", padx=18, pady=(0, 16))

        self.create_info_line(info_frame, self.origin_info_var)
        self.create_info_line(info_frame, self.dest_info_var)
        self.create_info_line(info_frame, self.distance_info_var)

        self.route_var.trace_add("write", self.update_route_info)

    def create_button_card(self, parent):
        button_frame = tk.Frame(parent, bg=self.colors["bg"])
        button_frame.pack(fill="x", pady=(0, 14))

        predict_button = ttk.Button(
            button_frame,
            text="Predict delay probability",
            command=self.run_prediction,
            style="Primary.TButton"
        )
        predict_button.pack(fill="x", ipady=5)

    def create_prediction_card(self, parent):
        prediction_card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        prediction_card.pack(fill="both", expand=True, pady=(0, 14))

        top = tk.Frame(prediction_card, bg=self.colors["surface"])
        top.pack(fill="x", padx=20, pady=(18, 8))

        tk.Label(
            top,
            text="Prediction Result",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"]
        ).pack(side="left")

        self.prediction_result_var = tk.StringVar(value="No prediction yet")

        self.prediction_badge = tk.Label(
            top,
            textvariable=self.prediction_result_var,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["neutral"],
            fg=self.colors["text"],
            padx=14,
            pady=7
        )
        self.prediction_badge.pack(side="right")

        tk.Label(
            prediction_card,
            text="Calibrated probabilities",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        ).pack(anchor="w", padx=20, pady=(6, 10))

        self.on_time_prob_var = tk.StringVar(value="On time: -")
        self.delay_prob_var = tk.StringVar(value="Delay: -")
        self.long_delay_prob_var = tk.StringVar(value="Long delay: -")

        self.on_time_bar = self.create_probability_row(
            prediction_card,
            "On time",
            self.on_time_prob_var,
            "Green.Horizontal.TProgressbar"
        )

        self.delay_bar = self.create_probability_row(
            prediction_card,
            "Delay",
            self.delay_prob_var,
            "Orange.Horizontal.TProgressbar"
        )

        self.long_delay_bar = self.create_probability_row(
            prediction_card,
            "Long delay",
            self.long_delay_prob_var,
            "Red.Horizontal.TProgressbar"
        )

        self.prediction_explanation_var = tk.StringVar(
            value="Select flight information and run the model to display calibrated probabilities."
        )

        explanation_box = tk.Frame(
            prediction_card,
            bg=self.colors["surface_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        explanation_box.pack(fill="x", padx=20, pady=(18, 18))

        tk.Label(
            explanation_box,
            textvariable=self.prediction_explanation_var,
            font=("Segoe UI", 10),
            bg=self.colors["surface_alt"],
            fg=self.colors["muted"],
            justify="left",
            wraplength=630
        ).pack(anchor="w", padx=14, pady=12)

    def create_project_summary_card(self, parent):
        summary_card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        summary_card.pack(fill="x")

        tk.Label(
            summary_card,
            text="Dataset and Model Summary",
            font=("Segoe UI", 15, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"]
        ).pack(anchor="w", padx=20, pady=(16, 10))

        summary = self.predictor.get_summary()

        total_rows = summary.get("total_rows", 0)
        train_rows = summary.get("train_rows", 0)
        val_rows = summary.get("validation_rows", 0)
        test_rows = summary.get("test_rows", 0)

        accuracy = summary.get("calibrated_accuracy")
        macro_f1 = summary.get("calibrated_macro_f1")
        weighted_f1 = summary.get("calibrated_weighted_f1")

        metrics_frame = tk.Frame(summary_card, bg=self.colors["surface"])
        metrics_frame.pack(fill="x", padx=20, pady=(0, 18))

        self.create_metric_tile(
            metrics_frame,
            "Total records",
            f"{total_rows:,}" if total_rows else "-",
            0,
            0
        )

        self.create_metric_tile(
            metrics_frame,
            "Train / Val / Test",
            f"{train_rows:,} / {val_rows:,} / {test_rows:,}" if total_rows else "-",
            0,
            1
        )

        self.create_metric_tile(
            metrics_frame,
            "Features",
            f"{summary.get('categorical_features', 0)} cat + {summary.get('numerical_features', 0)} num",
            0,
            2
        )

        self.create_metric_tile(
            metrics_frame,
            "Classes",
            str(summary.get("num_classes", "-")),
            1,
            0
        )

        self.create_metric_tile(
            metrics_frame,
            "Calibrated Accuracy",
            f"{accuracy:.4f}" if accuracy is not None else "-",
            1,
            1
        )

        self.create_metric_tile(
            metrics_frame,
            "Macro F1 / Weighted F1",
            f"{macro_f1:.4f} / {weighted_f1:.4f}" if macro_f1 is not None else "-",
            1,
            2
        )

    def create_metric_tile(self, parent, label, value, row, col):
        tile = tk.Frame(
            parent,
            bg=self.colors["surface_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1
        )
        tile.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        parent.grid_columnconfigure(col, weight=1)

        tk.Label(
            tile,
            text=label,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["surface_alt"],
            fg=self.colors["muted"]
        ).pack(anchor="w", padx=12, pady=(10, 2))

        tk.Label(
            tile,
            text=value,
            font=("Segoe UI", 13, "bold"),
            bg=self.colors["surface_alt"],
            fg=self.colors["text"]
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def create_info_line(self, parent, variable):
        tk.Label(
            parent,
            textvariable=variable,
            font=("Segoe UI", 10),
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            anchor="w",
            justify="left",
            wraplength=440
        ).pack(fill="x", pady=3)

    def create_probability_row(self, parent, title, variable, bar_style):
        wrapper = tk.Frame(parent, bg=self.colors["surface"])
        wrapper.pack(fill="x", padx=20, pady=8)

        top = tk.Frame(wrapper, bg=self.colors["surface"])
        top.pack(fill="x")

        tk.Label(
            top,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"]
        ).pack(side="left")

        tk.Label(
            top,
            textvariable=variable,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["muted"]
        ).pack(side="right")

        bar = ttk.Progressbar(
            wrapper,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            style=bar_style
        )
        bar.pack(fill="x", pady=(6, 0), ipady=4)

        return bar

    def add_combobox(self, parent, label, variable, values, row, width=35):
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            anchor="w"
        ).grid(row=row, column=0, sticky="w", padx=8, pady=7)

        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=width,
            state="readonly"
        )
        combo.grid(row=row, column=1, padx=8, pady=7, sticky="w")

        return combo

    def set_defaults(self):
        if self.carrier_options:
            self.carrier_var.set(self.carrier_options[0])

        if self.route_options:
            self.route_var.set(self.route_options[0])

        self.month_var.set("11")
        self.day_var.set("1 - Monday")
        self.dep_time_var.set("06:00")
        self.arr_time_var.set("09:00")

    def get_selected_route_row(self):
        selected_display_route = self.route_var.get()

        route_row = self.route_lookup[
            self.route_lookup["display_route"] == selected_display_route
        ]

        if route_row.empty:
            return None

        return route_row.iloc[0]

    def update_route_info(self, *args):
        route_row = self.get_selected_route_row()

        if route_row is None:
            self.origin_info_var.set("Origin: -")
            self.dest_info_var.set("Destination: -")
            self.distance_info_var.set("Distance: -")
            return

        self.origin_info_var.set(
            f"Origin: {route_row['ORIGIN']} | "
            f"{route_row['ORIGIN_CITY_NAME']} | "
            f"{route_row['ORIGIN_STATE_ABR']}"
        )

        self.dest_info_var.set(
            f"Destination: {route_row['DEST']} | "
            f"{route_row['DEST_CITY_NAME']} | "
            f"{route_row['DEST_STATE_ABR']}"
        )

        self.distance_info_var.set(
            f"Distance: {float(route_row['DISTANCE']):,.1f} miles"
        )

    def run_prediction(self):
        try:
            route_row = self.get_selected_route_row()

            if route_row is None:
                messagebox.showerror("Input error", "Please select a valid route.")
                return

            if not self.carrier_var.get():
                messagebox.showerror("Input error", "Please select a carrier.")
                return

            input_data = {
                "carrier": self.carrier_var.get(),
                "origin": route_row["ORIGIN"],
                "origin_state": route_row["ORIGIN_STATE_ABR"],
                "dest": route_row["DEST"],
                "dest_state": route_row["DEST_STATE_ABR"],
                "distance": float(route_row["DISTANCE"]),
                "month": int(self.month_var.get()),
                "day_of_week": int(self.day_var.get().split(" - ")[0]),
                "departure_time": self.dep_time_var.get(),
                "arrival_time": self.arr_time_var.get()
            }

            prediction = self.predictor.predict(input_data)

            self.display_prediction(prediction, input_data)

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    def display_prediction(self, prediction, input_data):
        predicted_label = prediction["predicted_label"]
        calibrated_probs = prediction["calibrated_probabilities"]

        self.prediction_result_var.set(f"Predicted: {predicted_label}")

        if predicted_label.lower() == "on time":
            self.prediction_badge.config(
                bg=self.colors["light_success"],
                fg=self.colors["success"]
            )
        elif predicted_label.lower() == "delay":
            self.prediction_badge.config(
                bg=self.colors["light_warning"],
                fg=self.colors["warning"]
            )
        else:
            self.prediction_badge.config(
                bg=self.colors["light_danger"],
                fg=self.colors["danger"]
            )

        on_time = calibrated_probs.get("On time", 0) * 100
        delay = calibrated_probs.get("Delay", 0) * 100
        long_delay = calibrated_probs.get("Long delay", 0) * 100

        self.on_time_prob_var.set(f"{on_time:.2f}%")
        self.delay_prob_var.set(f"{delay:.2f}%")
        self.long_delay_prob_var.set(f"{long_delay:.2f}%")

        self.on_time_bar["value"] = on_time
        self.delay_bar["value"] = delay
        self.long_delay_bar["value"] = long_delay

        route = f"{input_data['origin']} -> {input_data['dest']}"
        date_info = f"Month {input_data['month']}, day {input_data['day_of_week']}"
        schedule = f"{input_data['departure_time']} to {input_data['arrival_time']}"

        explanation = (
            f"Input summary: carrier {input_data['carrier']}, route {route}, "
            f"{date_info}, scheduled time {schedule}, "
            f"distance {float(input_data['distance']):,.1f} miles. "
            f"The prediction uses calibrated probabilities from the trained TabTransformer."
        )

        self.prediction_explanation_var.set(explanation)

        print("\nPrediction result:")
        print("Predicted class:", predicted_label)
        print("Calibrated probabilities:", calibrated_probs)


def main():
    base_dir = Path(__file__).resolve().parent.parent

    route_lookup_path = (
        base_dir
        / "data"
        / "generated"
        / "processed"
        / "route_lookup.csv"
    )

    if not route_lookup_path.exists():
        raise FileNotFoundError(
            f"Missing route lookup file: {route_lookup_path}\n"
            "Create it first from Notebook 06."
        )

    route_lookup = pd.read_csv(route_lookup_path)

    predictor = FlightDelayPredictor(base_dir)

    root = tk.Tk()
    FlightDelayApp(root, predictor, route_lookup)
    root.mainloop()


if __name__ == "__main__":
    main()