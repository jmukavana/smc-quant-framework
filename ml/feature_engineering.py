import pandas as pd
import numpy as np


class MLFeatureEngineer:
    """
    Phase 8A: Machine Learning Feature Pipeline
    Transforms raw and enriched data into a normalized feature matrix and generates target variables.
    """

    def __init__(self, forward_look: int = 10, r_target: float = 2.0):
        """
        :param forward_look: How many future candles to look ahead for the target.
        :param r_target: The risk multiplier we want to achieve (e.g. 2R).
        """
        self.forward_look = forward_look
        self.r_target = r_target

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and normalizes features from Phase 1, 2, and 3 data.
        """
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        # --- 1. Momentum & Volatility Features ---
        df['TR'] = df[['High']].max(axis=1) - df[['Low']].min(axis=1)
        features['ATR_Normalized'] = df['TR'].rolling(14).mean() / df['Close']
        features['Volume_Delta'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # --- 2. SMC Boolean to Numeric ---
        features['BOS_Bull'] = df.get('BOS_Bull', False).astype(int)
        features['BOS_Bear'] = df.get('BOS_Bear', False).astype(int)
        features['CHoCH_Bull'] = df.get('CHoCH_Bull', False).astype(int)
        features['Sweep_Bull'] = df.get('Sweep_Bull', False).astype(int)
        features['FVG_Fill_Prob'] = df.get('FVG_Fill_Prob', 0.0)

        # --- 3. Distance Features (Crucial for ML) ---
        # Instead of raw prices, ML needs *distances* (percentages)
        if 'Active_OB_Bull' in df.columns:
            features['Dist_to_OB_Bull'] = (df['Close'] - df['Active_OB_Bull']) / df['Close']
            features['Dist_to_OB_Bull'] = features['Dist_to_OB_Bull'].fillna(0)

        if 'Last_Swing_High' in df.columns:
            features['Dist_to_Swing_High'] = (df['Last_Swing_High'] - df['Close']) / df['Close']

        # --- 4. Rolling Counters (Memory) ---
        # How many BOS happened in the last 20 bars?
        features['BOS_Count_20'] = features['BOS_Bull'].rolling(20).sum() - df.get('BOS_Bear', False).astype(
            int).rolling(20).sum()

        # Handle NaNs created by rolling windows
        features = features.fillna(0)
        return features

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the binary classification target.
        Target_Long = 1 IF the maximum price in the next 'forward_look' candles
        exceeds 2x the current ATR (simulating a 1:2 RR win).
        """
        df = df.copy()

        # Approximate Risk (1R) as the current 14-period ATR
        df['TR'] = df[['High']].max(axis=1) - df[['Low']].min(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()

        # Look ahead for max high and min low
        df['Future_Max'] = df['High'].shift(-self.forward_look).rolling(self.forward_look).max()
        df['Future_Min'] = df['Low'].shift(-self.forward_look).rolling(self.forward_look).min()

        # Target 1: Did we hit a 2R profit before a 1R loss? (Simplified heuristic)
        # Note: In a true tick-level backtest we'd use paths, but for OHLCV ML target generation this is standard.
        profit_target = df['Close'] + (df['ATR'] * self.r_target)
        stop_loss = df['Close'] - df['ATR']

        # Hit target AND didn't hit SL first (Approximation using min/max bounds)
        hit_tp = df['Future_Max'] >= profit_target
        hit_sl = df['Future_Min'] <= stop_loss

        # Target Long: 1 if TP hit and SL not hit (or assuming TP hit first if both breached in the window)
        # A more rigorous approach requires intra-candle tick data, but this serves the ML objective.
        df['Target_Long'] = (hit_tp & ~hit_sl).astype(int)

        return df[['Target_Long']]

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        """ Master pipeline to generate X and y. """
        X = self.create_features(df)
        if not is_training:
            return X, None

        y = self.create_targets(df)

        # Drop the last 'forward_look' rows because their targets are NaN/Unknown
        X = X.iloc[:-self.forward_look]
        y = y.iloc[:-self.forward_look]

        return X, y