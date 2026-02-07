# Credit Card Fraud Detection with Deep Learning (MLP vs Autoencoder)

## Project Description
This project focuses on detecting fraudulent credit card transactions using deep learning techniques. Given the extreme class imbalance in the dataset (fraud accounts for only 0.17% of all transactions), standard metrics like Accuracy are misleading. Therefore, this study employs the Area Under the Precision-Recall Curve (AUPRC) as the primary evaluation metric.

The project conducts a comparative analysis of two distinct neural network architectures implemented in PyTorch:
1. **Supervised Advanced MLP:** A deep feed-forward neural network trained with supervision, utilizing Weighted BCE Loss to handle class imbalance effectively.
2. **Unsupervised Tuned Autoencoder:** An anomaly detection model trained exclusively on legitimate transactions. It identifies fraud by measuring the reconstruction error, assuming that fraudulent patterns will yield higher errors than normal ones.
3. **Boosting Algorithms** Tree-based gradient boosting models that build an ensemble of decision trees.
The framework **Optuna** is utilized for automated hyperparameter optimization, tuning critical parameters such as the number of layers, hidden units, learning rate, and bottleneck size.

## Installation and Usage

### 1. Clone Repository
Clone the project to your local machine:
```bash
git clone https://github.com/your-username/fraud-detection-project.git
```
### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using the requirements file:
```bash
pip install -r requirements.txt
```
### 3. Data Preparation
1. Download the dataset `creditcard.csv` from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/whenamancodes/fraud-detection).
2. Place the `creditcard.csv`file in the root directory of the project.

### 4. Running the Project
The project is implemented as a Jupyter Notebook. Open and run the notebook:
jupyter notebook Untitled1.ipynb

The notebook contains sequential cells for data preprocessing, hyperparameter tuning via Optuna, final model training, and results visualization.

## Data Overview
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
- **Total Transactions:** 284,807
- **Fraud Cases (Class 1):** 492 (0.172%)
- **Features:** `Time`, `Amount`, and 28 anonymized features (`V1`-`V28`) obtained via PCA.
- **Preprocessing:** A strict time-based split (80% Train, 20% Test) is used to prevent data leakage. All features, including V1-V28, are normalized to the [0, 1] range using `MinMaxScaler` to ensure stability for the Autoencoder.

## Methodology

### Supervised MLP (Multi-Layer Perceptron)
- **Architecture:** A dynamic deep network (2–4 layers) featuring Batch Normalization, LeakyReLU activation, and Dropout for regularization.
- **Training Strategy:** Utilizes Class Weighting in the loss function to heavily penalize missing fraud cases (False Negatives).

### Gradient Boosting with XGBoost
- **Architecture:** Tree-based gradient boosting model that builds an ensemble of decision trees, each correcting the residual errors of the previous ones.
- **Optimization:** Hyperparameters such as max_depth, learning_rate, n_estimators, subsample, colsample_bytree, gamma, and scale_pos_weight are tuned with Optuna to maximize AUPRC on the validation set under heavy class imbalance.

### Gradient Boosting with LightGBM
- **Architecture:** Histogram-based gradient boosting decision tree model optimized for speed and memory efficiency, using leaf-wise growth with depth constraints.
- **Optimization:** Optuna searches over num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, regularization parameters, learning_rate, n_estimators, and scale_pos_weight to achieve the best precision–recall trade-off on the fraud class.

### Unsupervised Autoencoder
- **Concept:** The model learns a compressed representation of normal transactions. During inference, transactions with a reconstruction error exceeding a dynamically calculated threshold are flagged as fraud.
- **Optimization:** Optuna tunes the bottleneck size (latent space dimension) and selects the optimal reconstruction loss function (L1 vs. MSE vs. BCE), while early stopping and learning rate scheduling are used to prevent overfitting on normal data.


## Experimental Results

The models were evaluated based on AUPRC, which is robust to class imbalance.

| Model           | AUPRC | Analysis |
| :--------------| :-----| :--------|
| **Advanced MLP** | **0.769** | Demonstrated superior performance by leveraging labeled data. Achieved high recall while maintaining good precision. |
| **XGBoost**      | **0.804** | Achieved the best overall ranking quality on the fraud class, providing the highest AUPRC and strong precision–recall balance on the test set. |
| **LightGBM**     | **0.74** | Performed competitively with MLP, slightly below XGBoost in AUPRC, while maintaining stable precision and recall under severe class imbalance. |
| **Autoencoder**  | 0.17| Effectively functioned as an anomaly filter but yielded a higher False Positive rate compared to the supervised approaches. |

### Visualization
The project output includes:
1. **Precision-Recall Curves:** A comparative plot showing the trade-off between precision and recall for both models.
2. **Confusion Matrices:** Heatmaps visualizing the classification performance, highlighting True Positives and False Negatives.

## Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch
- **Optimization:** Optuna
- **Data Processing:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn

## Authors
- Nurbek Seilbek
- Danial Yermekov



