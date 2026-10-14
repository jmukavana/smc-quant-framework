import pandas as pd
import numpy as np


class MarketStructureEngine:
    """
        Phase 1: Institutional Market Structure Engine
        Detects Swing Highs/Lows, Impulsive/Corrective Waves, and categorizes Trends.
        Agnostic to timeframe and asset class (Crypto, Forex, Equities).
        """

    def __init__(self, swing_window: int = 5):
        """
        :param swing_window: Number of bars to the left and right required to confirm a swing pivot.
                             Default 5 is standard for institutional multi-timeframe analysis.
        """
        self.swing_window = swing_window
    def _detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized detection of structural pivot points.
        Uses rolling windows to prevent look-ahead bias in live execution.
        """
        df = df.copy()

        # High and Low rolling windows
        roll_high = df['High'].rolling(window=2 * self.swing_window + 1, center=True).max()
        roll_low = df['Low'].rolling(window=2 * self.swing_window + 1, center=True).min()

        # Identify pivots (Boolean masks)
        # In a live setting, a pivot at index i is only known at index i + swing_window
        df['is_Swing_High'] = (df['High'] == roll_high)
        df['is_Swing_Low'] = (df['Low'] == roll_low)

        # Extract values where True, else NaN
        df['Swing_High_Price'] = np.where(df['is_Swing_High'], df['High'], np.nan)
        df['Swing_Low_Price'] = np.where(df['is_Swing_Low'], df['Low'], np.nan)

        # Forward fill previous swing levels for structural comparison
        df['Last_Swing_High'] = df['Swing_High_Price'].ffill()
        df['Last_Swing_Low'] = df['Swing_Low_Price'].ffill()

        return df

    def _classify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies HH, HL, LH, LL to determine the micro and macro trend.
        """
        # Get purely the historical pivots in order
        highs = df[df['is_Swing_High']]['High']
        lows = df[df['is_Swing_Low']]['Low']

        # Determine Higher Highs, Lower Highs, etc.
        df['Structure_High'] = np.nan
        df['Structure_Low'] = np.nan

        # Logic for Highs: HH or LH
        prev_high = highs.shift(1)
        df.loc[highs.index, 'Structure_High'] = np.where(highs > prev_high, 'HH', 'LH')

        # Logic for Lows: HL or LL
        prev_low = lows.shift(1)
        df.loc[lows.index, 'Structure_Low'] = np.where(lows > prev_low, 'HL', 'LL')

        df['Structure_High'] = df['Structure_High'].ffill()
        df['Structure_Low'] = df['Structure_Low'].ffill()

        return df

    def _determine_waves_and_trend(self, df: pd.DataFrame) -> dict:
        """
        Analyzes the compiled structural points to output institutional trend strength
        and impulsive/corrective wave states.
        """
        # Get the most recent confirmed structural points
        recent_high_struct = df['Structure_High'].iloc[-1]
        recent_low_struct = df['Structure_Low'].iloc[-1]

        last_high_price = df['Last_Swing_High'].iloc[-1]
        last_low_price = df['Last_Swing_Low'].iloc[-1]

        # Calculate current wave distance
        current_price = df['Close'].iloc[-1]

        # Determine Trend Direction and Strength
        trend_direction = "Sideways"
        trend_strength = "Weak"

        if recent_high_struct == 'HH' and recent_low_struct == 'HL':
            trend_direction = "Bullish"
            trend_strength = "Strong"
        elif recent_high_struct == 'HH' and recent_low_struct == 'LL':
            trend_direction = "Bullish"
            trend_strength = "Weak"  # Expanding megaphone/loss of momentum
        elif recent_high_struct == 'LH' and recent_low_struct == 'LL':
            trend_direction = "Bearish"
            trend_strength = "Strong"
        elif recent_high_struct == 'LH' and recent_low_struct == 'HL':
            trend_direction = "Bearish"
            trend_strength = "Weak"  # Contracting triangle/loss of momentum
        else:
            trend_direction = "Sideways"
            trend_strength = "Neutral"

        # Determine if current wave is Impulsive or Corrective
        is_impulsive = False
        if trend_direction == "Bullish" and current_price > last_low_price:
            # Moving up in a bullish trend
            is_impulsive = True
        elif trend_direction == "Bearish" and current_price < last_high_price:
            # Moving down in a bearish trend
            is_impulsive = True

        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "current_wave": "Impulsive" if is_impulsive else "Corrective",
            "swing_highs": {"price": float(last_high_price), "type": recent_high_struct},
            "swing_lows": {"price": float(last_low_price), "type": recent_low_struct}
        }

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Master Pipeline for Phase 1.
        Takes standard OHLCV dataframe, processes structure, returns state dict.
        """
        if len(df) < self.swing_window * 3:
            raise ValueError("Insufficient data to calculate institutional market structure.")

        df = self._detect_pivots(df)
        df = self._classify_market_structure(df)

        # We can attach the enriched dataframe back to the engine for later phases
        self.enriched_data = df

        # Return cleanly formatted state for Machine Learning / Setup triggers
        state = self._determine_waves_and_trend(df)
        return state

    if __name__ == "__main__":
        # Quick Test Execution Block
        print("Testing Phase 1: Market Structure Engine...")

        # Generate Dummy OHLCV data representing a bullish trend
        np.random.seed(42)
        base_price = 100
        prices = [base_price + (i * 1.5) + np.random.normal(0, 2) for i in range(100)]

        mock_df = pd.DataFrame({
            'Open': prices,
            'High': [p + 2 for p in prices],
            'Low': [p - 2 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100, 1000, 100)
        })

        engine = MarketStructureEngine(swing_window=5)
        market_state = engine.analyze(mock_df)

        import json
        print(json.dumps(market_state, indent=4))



