import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import os


class MLPredictor:
    """
    Phase 8B: XGBoost Model Training and Inference Pipeline
    Trains a classification model to predict the probability of a successful Long setup.
    """

    def __init__(self, model_path: str = "ml/models/xgboost_long.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Trains the XGBoost Classifier.
        """
        print(f"Starting ML Training on {len(X)} samples...")

        # Split data (Chronological split to prevent data leakage/lookahead bias)
        # In time-series, NEVER use random train_test_split. The past must predict the future.
        split_idx = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Ensure target is 1D array
        y_train = y_train['Target_Long'].values
        y_test = y_test['Target_Long'].values

        # Initialize Model
        # Scale_pos_weight is crucial because trading targets are heavily imbalanced
        # (Winning 2R trades happen maybe 30% of the time).
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # Fit model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=20
        )

        # Evaluation
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)

        print("\n--- Model Evaluation ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f} (Crucial metric: When it says BUY, how often is it right?)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Ensure model directory exists and save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No trained model found at {self.model_path}")

    def predict_probability(self, current_features: pd.DataFrame) -> float:
        """
        Live inference. Returns the probability [0.0 - 1.0] of a successful Long.
        """
        if self.model is None:
            self.load_model()

        # Get the latest row of features
        latest_X = current_features.iloc[[-1]]

        # Predict probability of class 1 (Target Hit)
        prob = self.model.predict_proba(latest_X)[0][1]
        return float(prob)


if __name__ == "__main__":
    from feature_engineering import MLFeatureEngineer

    print("Testing Phase 8: ML Pipeline...")

    # 1. Generate Mock Market Data
    np.random.seed(42)
    rows = 2000
    mock_df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, rows),
        'High': np.random.uniform(200, 210, rows),
        'Low': np.random.uniform(90, 100, rows),
        'Close': np.random.uniform(100, 200, rows),
        'Volume': np.random.randint(100, 1000, rows),
        'BOS_Bull': np.random.choice([True, False], rows, p=[0.1, 0.9]),
        'CHoCH_Bull': np.random.choice([True, False], rows, p=[0.05, 0.95]),
        'Active_OB_Bull': np.random.uniform(95, 105, rows)
    })

    # 2. Extract Features and Targets
    engineer = MLFeatureEngineer(forward_look=10, r_target=2.0)
    X, y = engineer.prepare_data(mock_df, is_training=True)

    # 3. Train Model
    predictor = MLPredictor(model_path="models/xgboost_long.pkl")
    predictor.train(X, y)

    # 4. Live Inference Test
    latest_prob = predictor.predict_probability(X)
    print(f"\nLive Inference Probability of Success: {latest_prob * 100:.2f}%")
