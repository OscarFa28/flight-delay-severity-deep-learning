This project presents a deep learning pipeline for multiclass flight delay severity prediction using real-world 2025 flight records from the U.S. Bureau of Transportation Statistics (BTS). The task is formulated as a temporal classification problem with three classes: On time, Delay, and Long delay.

The project compares three PyTorch models: an MLP with categorical embeddings, a Wide & Deep Neural Network, and a calibrated TabTransformer. A temporal train/validation/test split is used to simulate future flight prediction and avoid inflated random-split results. The final model is selected using Macro F1-score due to class imbalance.

The repository also includes a Python graphical interface that loads the trained TabTransformer and estimates calibrated delay probabilities for selected flight information.
