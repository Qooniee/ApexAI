# Acknowledgments

ApexAI utilizes and builds upon numerous high-quality open-source libraries, with full compliance to all applicable licenses (MIT, BSD, Apache 2.0, PostgreSQL License, etc.). The integration of these tools forms the robust, containerized MLOps foundation described in our project.

---

## 1. Core Frameworks and Deep Learning Engine

These libraries constitute the primary computational and model development backbone of the platform.

- **PyTorch / TorchVision / TorchAudio**: Core deep learning framework used for all model training (LSTM, GRU, Transformer, Informer) and GPU acceleration. (BSD License)

- **Optuna / Optuna-Dashboard**: Provides the engine for Bayesian Optimization-based Neural Architecture Search (BO-NAS) and Hyperparameter Optimization (HPO). (MIT License)

- **MLflow**: Used for all experiment tracking, metadata logging, and model artifact management, ensuring complete reproducibility (MLOps compliance). (Apache License 2.0)

- **Hydra-Core / OmegaConf**: Enables configuration management for repeatable experiments across different models and datasets. (MIT License)

---

## 2. Scientific Computing and Data Processing

These components are essential for data pipeline automation and feature engineering.

- **NumPy / Pandas / SciPy**: Fundamental libraries for mathematical operations, matrix handling, and data manipulation within the data pipeline. (BSD License)

- **Scikit-learn**: Utilized for data preprocessing, utility functions, and performance metrics (e.g., F1-score calculation). (BSD License)

- **Polars / DuckDB**: High-performance libraries used for efficient handling of large-scale time-series telemetry data during preprocessing and feature extraction. (MIT / Apache License 2.0)

---

## 3. Deployment and Infrastructure

These tools provide the core containerization, storage, and database management necessary for the on-premise MLOps environment.

- **Docker / Docker Compose**: Container platform used for one-command deployment and service orchestration. (Apache License 2.0)

- **PostgreSQL / Psycopg2**: Used as the backend database for storing MLflow and Optuna metadata. (PostgreSQL License)

- **MinIO / Boto3**: Used for S3-compatible artifact storage (for trained models and logs). (Apache License 2.0)

- **Streamlit**: Powers the real-time inference simulator (frontend UI). (Apache License 2.0)

---

## 4. Visualization and Utilities

These tools enhance user experience through visualization and progress tracking.

- **Matplotlib / Seaborn / Plotly**: Used for data visualization and graphing within EDA notebooks and the Streamlit simulator. (PSF-based / BSD / MIT License)

- **tqdm**: Used for visualizing progress during long-running tasks like training and HPO. (MIT / MPL License)

---

## License Compliance Statement

All dependencies listed above are used in accordance with their respective open-source licenses. ApexAI respects and adheres to all license terms, including attribution requirements, distribution conditions, and warranty disclaimers.

For detailed license information of each library, please refer to their respective repositories:
- PyTorch: https://github.com/pytorch/pytorch
- Optuna: https://github.com/optuna/optuna
- MLflow: https://github.com/mlflow/mlflow
- Streamlit: https://github.com/streamlit/streamlit
- (And others listed above)

---

## Project License

ApexAI itself is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Disclaimer**: This project was developed for the Hack the Track hackathon. All third-party software components are acknowledged and credited in accordance with their respective license requirements.
