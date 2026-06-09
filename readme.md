# Flight Delay Severity Prediction with Deep Learning

This repository contains a Deep Learning project for multiclass flight delay severity prediction using real-world flight records from the U.S. Bureau of Transportation Statistics (BTS) TranStats Airline On-Time Performance Data.

The project uses flight records from **2023, 2024, and 2025**, resulting in approximately **21 million raw records**. The final model-ready dataset contains **20,588,160 flights** after cleaning and feature preparation.

The main objective is to predict flight delay severity using only **pre-flight or schedule-based information**, while avoiding direct data leakage from post-flight delay variables.

---

## Project Overview

Flight delays affect passengers, airlines, airport operations, crew scheduling, and downstream flight planning. Instead of treating the problem as a simple binary delayed/not delayed task, this project predicts delay severity using three classes:

| Class | Label      | Definition                          |
| ----- | ---------- | ----------------------------------- |
| 0     | On time    | Arrival delay <= 15 minutes         |
| 1     | Delay      | Arrival delay from 16 to 90 minutes |
| 2     | Long delay | Arrival delay > 90 minutes          |

The original project idea considered a 5-class target, but the very rare long-delay categories created severe class imbalance. The target was redefined into 3 classes to create a more stable and meaningful classification problem.

The final task is intentionally designed as an **early delay-risk estimation problem**. The model does not use real-time information such as weather, airport congestion, maintenance events, or aircraft rotation. Instead, it uses stable information that is usually known before the flight date, such as carrier, route, origin, destination, distance, scheduled departure time, month, and day of week.

---

## Approach Rationale

Many flight delay prediction systems include external operational data, especially weather. This can improve predictive performance because weather is one of the strongest causes of flight disruptions. However, this project follows a different approach.

The goal is to estimate delay risk using only information that is known in advance. This is relevant because passengers often buy flights weeks or months before traveling. At that moment, exact weather conditions, airport congestion, maintenance issues, air traffic control restrictions, and aircraft rotation status are not available. Some of those variables would require their own forecasting models.

For that reason, this project focuses on fixed or schedule-based features. The model is not intended to replace real-time airline operations systems. Instead, it is designed to provide an early probability estimate that can help compare flight options, evaluate short connection risks, and understand historical delay patterns associated with specific routes, carriers, months, and schedules.

This design makes the problem harder, especially for the `Long delay` class, but it also makes the experiment different from approaches that rely heavily on external or near-real-time data.

---

## Dataset

The dataset comes from the **U.S. Bureau of Transportation Statistics (BTS) TranStats Airline On-Time Performance Data**.

Dataset source:

https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD&DB_URL=

Although BTS provides records from 1987 to 2025, this project uses only the years **2023, 2024, and 2025**.

This range was selected for three main reasons:

1. **Dataset size**: approximately 21 million raw records were already enough for large-scale Deep Learning experiments.
2. **COVID-19 abnormality**: including 2020, 2021, or part of 2022 could introduce unusual aviation patterns caused by the COVID-19 period.
3. **Current relevance**: very old records may represent airline reliability, airport operations, route availability, and operational behavior that are less aligned with current aviation conditions.

The raw input consists of **36 monthly CSV files**.

---

## Dataset Availability

The original dataset is **not included in this repository** because of its size. The complete raw dataset contains approximately 21 million records across 36 monthly files, which is too large for a regular GitHub repository.

To reproduce the project, download the monthly CSV files from BTS TranStats and place them inside:

```text
data/raw/
```

Expected data range:

```text
2023: January to December
2024: January to December
2025: January to December
```

After placing the files in `data/raw/`, run the notebooks in order. The preprocessing notebooks will regenerate the Parquet files, processed NumPy datasets, encoders, scalers, and model-ready artifacts.

---

## Dataset Size

| Dataset Version     |    Records |
| ------------------- | ---------: |
| Raw unified dataset |       ~21M |
| Model-ready dataset | 20,588,160 |
| Training set        | 14,411,712 |
| Validation set      |  3,088,224 |
| Test set            |  3,088,224 |

---

## Data Split

The project uses a **stratified random split**:

| Split      | Percentage |
| ---------- | ---------: |
| Train      |        70% |
| Validation |        15% |
| Test       |        15% |

The split is stratified by the target class to preserve class proportions across all subsets. This is important because the dataset is imbalanced and most flights belong to the `On time` class.

Since the dataset includes three complete years of records, the stratified split also helps represent different seasons and operational periods across training, validation, and test sets.

---

## Feature Engineering

The model uses only pre-flight or schedule-based information to avoid data leakage.

### Categorical Features

* Operating carrier
* Origin airport
* Origin state
* Destination airport
* Destination state
* Route
* Departure time block
* Arrival time block

### Numerical Features

* Distance
* Scheduled departure minutes of day
* Sine/cosine transformation of departure time
* Sine/cosine transformation of month
* Sine/cosine transformation of day of week
* Weekend indicator

Categorical variables are encoded as integer IDs and passed through embedding layers. Numerical variables are standardized using `StandardScaler`.

Post-flight variables such as actual arrival delay, actual departure delay, delay causes, taxi time, air time, and actual arrival/departure values are excluded from the model features.

---

## Leakage Prevention

A strict leakage prevention strategy is used. The model does not use variables that directly reveal the final delay or variables that would only be known after the flight happened.

Excluded variables include:

* Actual departure time
* Actual arrival time
* Departure delay
* Arrival delay
* Delay cause columns
* Taxi time
* Air time
* Carrier delay
* Weather delay
* NAS delay
* Security delay
* Late aircraft delay
* Cancellation information

This makes the task more difficult but also more realistic for early prediction.

---

## Models

Three PyTorch models are implemented and compared.

### 1. MLP with Embeddings

The baseline model uses embedding layers for categorical features and combines them with numerical variables through fully connected layers.

Architecture summary:

* Categorical embeddings
* Numerical feature concatenation
* Hidden layers: 256, 128, 64
* Dropout: 0.3
* Loss: CrossEntropyLoss with softened class weights
* Optimizer: AdamW

### 2. Wide & Deep Neural Network

This model combines two learning paths:

* A wide path for direct numerical signals
* A deep path for nonlinear interactions between categorical embeddings and numerical features

Architecture summary:

* Wide numerical path
* Deep embedding path
* Hidden layers: 384, 192, 96
* Dropout: 0.25
* Optimizer: AdamW

### 3. TabTransformer / FT-Transformer Style Model

The final and strongest model represents categorical and numerical features as tokens and processes them through a Transformer Encoder.

Architecture summary:

* Categorical tokens
* Numerical tokens
* CLS token
* Transformer Encoder
* CLS token + mean token pooling
* Calibration with validation-tuned class probability multipliers

Main configuration:

| Parameter             |    Value |
| --------------------- | -------: |
| d_model               |       96 |
| Attention heads       |        6 |
| Encoder layers        |        4 |
| Feedforward dimension |      384 |
| Transformer dropout   |     0.12 |
| Head hidden layers    | 384, 192 |
| Batch size            |     4096 |

---

## Final Results

Because the dataset is imbalanced, **Macro F1-score** is the main evaluation metric. Accuracy alone can be misleading because a model may achieve high accuracy by mostly predicting the majority class.

### Test Performance

| Model                     |   Accuracy | Balanced Accuracy |   Macro F1 | Weighted F1 |
| ------------------------- | ---------: | ----------------: | ---------: | ----------: |
| MLP with Embeddings       |     0.7162 |                 - |     0.4192 |      0.7179 |
| Wide & Deep               |     0.6539 |                 - |     0.4149 |      0.6817 |
| TabTransformer Raw        | **0.7344** |            0.4259 |     0.4309 |  **0.7284** |
| TabTransformer Calibrated |     0.7057 |        **0.4425** | **0.4346** |      0.7155 |

The **raw TabTransformer** achieved the highest accuracy and weighted F1-score. However, the **calibrated TabTransformer** achieved the best Macro F1-score and balanced accuracy, so it was selected as the final model.

Calibration improved the model's balance across classes, especially for delayed flights. This is important because `Long delay` is the hardest class and the most affected by missing external factors.

---

## Training Hardware

All experiments were trained on:

* HP OMEN 16
* NVIDIA GeForce RTX 4060 Laptop GPU
* Intel Core i9-13900HX
* 32 GB DDR5 RAM 5600 MHz

### Training Time

| Model               | Epochs | Training Time |
| ------------------- | -----: | ------------: |
| MLP with Embeddings |     10 |    ~59.90 min |
| Wide & Deep         |     10 |    ~54.49 min |
| TabTransformer      |     12 |   ~175.52 min |

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── monthly CSV files
│   └── generated/
│       ├── processed/
│       │   ├── train_data.npz
│       │   ├── val_data.npz
│       │   ├── test_data.npz
│       │   ├── metadata.json
│       │   ├── category_encoders.pkl
│       │   ├── numeric_scaler.pkl
│       │   └── route_lookup.csv
│       ├── flights_2023_2025_raw.parquet
│       ├── flights_2023_2025_clean_full.parquet
│       └── flights_2023_2025_model_base.parquet
│
├── models/
│   ├── mlp_embeddings_best.pt
│   ├── wide_deep_best.pt
│   └── tabtransformer_best.pt
│
├── notebooks/
│   ├── 01_data_preparation_eda.ipynb
│   ├── 02_feature_engineering_pytorch.ipynb
│   ├── 03_mlp_embeddings_pytorch.ipynb
│   ├── 04_wide_deep_pytorch.ipynb
│   ├── 05_tabtransformer_pytorch.ipynb
│   └── 06_results_analysis.ipynb
│
├── results/
│   ├── mlp_embeddings_results.json
│   ├── wide_deep_results.json
│   ├── tabtransformer_results.json
│   └── training histories and confusion matrices
│
├── paper_outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   └── predict_flight_delay_gui.py
│
├── README.md
└── requirements.txt
```

Large generated files may not be included in the repository. They can be regenerated by running the notebooks after downloading the original BTS CSV files.

---

## Notebooks

### 01 - Data Preparation and EDA

Loads and merges the 36 monthly CSV files from 2023 to 2025. It performs initial cleaning, removes cancelled and invalid flights, creates the target variable, and exports the clean datasets.

### 02 - Feature Engineering for PyTorch

Creates categorical encodings, numerical scaling, cyclic time features, class weights, and PyTorch-ready `.npz` files.

### 03 - MLP with Embeddings

Trains the baseline neural network model using categorical embeddings and numerical features.

### 04 - Wide & Deep Neural Network

Trains a hybrid architecture that combines a wide numerical path and a deep embedding-based path.

### 05 - TabTransformer

Trains the Transformer-based model and applies validation-based probability calibration.

### 06 - Results Analysis

Generates final result tables, figures, model comparisons, confusion matrices, and paper-ready outputs.

---

## Graphical Inference Interface

The repository includes a small Python GUI for testing the trained calibrated TabTransformer model.

The interface allows the user to select:

* Carrier
* Route
* Month
* Day of week
* Departure time
* Arrival time block

It then displays calibrated probabilities for:

* On time
* Delay
* Long delay

Run the GUI with:

```bash
python src/predict_flight_delay_gui.py
```

The GUI requires the trained model checkpoint and processed files generated by the notebooks.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the main dependencies manually:

```bash
pip install numpy pandas polars pyarrow scikit-learn scipy matplotlib seaborn tqdm joblib torch ipykernel notebook
```

---

## How to Reproduce the Project

1. Clone the repository:

```bash
git clone https://github.com/OscarFa28/flight-delay-severity-deep-learning.git
cd flight-delay-severity-deep-learning
```

2. Create and activate the virtual environment.

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the BTS TranStats Airline On-Time Performance monthly CSV files for:

```text
2023 January-December
2024 January-December
2025 January-December
```

5. Place all 36 CSV files inside:

```text
data/raw/
```

6. Run the notebooks in order:

```text
01_data_preparation_eda.ipynb
02_feature_engineering_pytorch.ipynb
03_mlp_embeddings_pytorch.ipynb
04_wide_deep_pytorch.ipynb
05_tabtransformer_pytorch.ipynb
06_results_analysis.ipynb
```

7. After training the models, run the GUI:

```bash
python src/predict_flight_delay_gui.py
```

---

## Important Notes

Large files such as raw CSVs, generated Parquet files, `.npz` processed datasets, and trained model checkpoints may be excluded from the repository due to storage limits.

Recommended files to exclude from Git:

```text
data/raw/
data/generated/
models/*.pt
*.npz
*.parquet
```

The repository is still reproducible because all generated artifacts can be recreated from the original BTS CSV files by running the notebooks in order.

---

## Limitations

The model only uses schedule-based and pre-flight information. It does not include external operational factors such as:

* Weather conditions
* Aircraft rotation
* Previous flight delay
* Airport congestion
* Maintenance events
* Air traffic control restrictions
* Real-time disruptions

These missing factors likely limit the prediction of the `Long delay` class. Therefore, the model should be interpreted as an early delay-risk estimator, not as a complete operational prediction system.

The results should be understood under this constraint. The model is not expected to predict every severe disruption because many severe disruptions are caused by events that are not visible from schedule-based information alone.

---

## Interpretability

The final model is based on a Transformer-style architecture. For tabular Transformers, attention-based analysis can be used as an interpretability tool.

A full attention-map visualization was not included in the final GUI. However, future work could extract attention weights from the Transformer Encoder to study which feature tokens influence delay predictions the most, such as route, carrier, time block, month, day of week, or distance.

Grad-CAM is not directly applicable to this project because the final model is not a convolutional neural network for image data. For this type of model, attention maps are a more appropriate interpretability direction.

---

## Future Work

Future work can extend the project in two directions.

### 1. Keeping the early-prediction approach

This direction preserves the original idea of using only stable information known before the flight date. Possible improvements include:

* Testing additional tabular architectures.
* Comparing against gradient boosting baselines such as XGBoost, LightGBM, or CatBoost.
* Running repeated experiments with different random seeds.
* Adding confidence intervals for the reported metrics.
* Performing feature ablation studies.
* Extracting attention maps from the TabTransformer.
* Testing cost-sensitive thresholds for missed long delays.
* Improving the GUI and exporting predictions to a report or dashboard.

### 2. Building an operational real-time version

This direction would no longer be a pure early-prediction model, but it could improve accuracy by including dynamic operational data such as:

* Weather forecasts or observed weather.
* Airport congestion indicators.
* Aircraft rotation information.
* Previous flight delay.
* Maintenance events.
* Air traffic control restrictions.
* Real-time disruption signals.

This second direction would likely improve `Long delay` prediction, but it would also require more data sources and would depend on information that may not be available far in advance.

---

## Author

Oscar Fabrizio de Alba Gutiérrez
Artificial Intelligence Engineering
Universidad Panamericana
Aguascalientes, Mexico
