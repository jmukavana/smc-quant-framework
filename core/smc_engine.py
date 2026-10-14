import pandas as pd
import numpy as np


class SMCEngine:
    """
    Phase 2: Smart Money Concepts Engine
    Detects BOS, CHoCH, Liquidity Sweeps, Order Blocks, and Fair Value Gaps (FVG).
    Requires a DataFrame enriched by Phase 1 (MarketStructureEngine).
    """

    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Fair Value Gaps (Imbalances).
        A Bullish FVG occurs when Candle 3's Low is higher than Candle 1's High.
        A Bearish FVG occurs when Candle 3's High is lower than Candle 1's Low.
        """
        df = df.copy()

        # Bullish FVG: Low of current candle > High of candle 2 periods ago
        df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(1))
        df['FVG_Bull_Size'] = np.where(df['FVG_Bull'], df['Low'] - df['High'].shift(2), 0)

        # Bearish FVG: High of current candle < Low of candle 2 periods ago
        df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(1))
        df['FVG_Bear_Size'] = np.where(df['FVG_Bear'], df['Low'].shift(2) - df['High'], 0)

        # Calculate Fill Probability (Heuristic based on size and current volatility/ATR)
        df['ATR_Approx'] = df['High'] - df['Low']
        df['FVG_Fill_Prob'] = np.where(
            df['FVG_Bull'] | df['FVG_Bear'],
            # Smaller FVGs relative to ATR have higher probability of being filled quickly
            np.clip(1 - (df[['FVG_Bull_Size', 'FVG_Bear_Size']].max(axis=1) / df['ATR_Approx'].rolling(14).mean()), 0.1,
                    0.99),
            0
        )
        return df

    @staticmethod
    def detect_bos_and_choch(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH).
        RULES: Candle must CLOSE above/below the previous structure. Wicks do not count.
        """
        df = df.copy()

        # Initialize columns
        df['BOS_Bull'] = False
        df['BOS_Bear'] = False
        df['CHoCH_Bull'] = False
        df['CHoCH_Bear'] = False
        df['CHoCH_Strength'] = 0.0  # 0 to 100 based on displacement volume

        # Bullish BOS: Close > Last Swing High
        bull_break = (df['Close'] > df['Last_Swing_High'].shift(1)) & \
                     (df['Close'].shift(1) <= df['Last_Swing_High'].shift(2))

        # Bearish BOS: Close < Last Swing Low
        bear_break = (df['Close'] < df['Last_Swing_Low'].shift(1)) & \
                     (df['Close'].shift(1) >= df['Last_Swing_Low'].shift(2))

        # Differentiate BOS (Continuation) from CHoCH (Reversal)
        # Using Phase 1 Trend logic approximations:
        # If we were making Lower Lows (Bear Trend) and we break a Swing High -> Bullish CHoCH
        is_bear_trend = df['Structure_High'].shift(1) == 'LH'
        is_bull_trend = df['Structure_Low'].shift(1) == 'HL'

        df['BOS_Bull'] = bull_break & is_bull_trend
        df['CHoCH_Bull'] = bull_break & is_bear_trend

        df['BOS_Bear'] = bear_break & is_bear_trend
        df['CHoCH_Bear'] = bear_break & is_bull_trend

        # CHoCH Strength Score (Displacement volume vs Average Volume)
        avg_vol = df['Volume'].rolling(20).mean()
        df['CHoCH_Strength'] = np.where(
            df['CHoCH_Bull'] | df['CHoCH_Bear'],
            np.clip((df['Volume'] / avg_vol) * 50, 0, 100),  # Cap at 100
            0
        )

        return df

    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Equal High/Low sweeps (External Liquidity).
        Rule: Price wicks beyond structure, but CLOSES inside structure.
        """
        df = df.copy()

        # Bullish Sweep (Sell-side liquidity taken):
        # Wick goes below Last Swing Low, but close is above it.
        df['Sweep_Bull'] = (df['Low'] < df['Last_Swing_Low'].shift(1)) & \
                           (df['Close'] > df['Last_Swing_Low'].shift(1))

        # Bearish Sweep (Buy-side liquidity taken):
        # Wick goes above Last Swing High, but close is below it.
        df['Sweep_Bear'] = (df['High'] > df['Last_Swing_High'].shift(1)) & \
                           (df['Close'] < df['Last_Swing_High'].shift(1))

        return df

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Institutional Order Blocks.
        Bullish OB: The last down-close candle before an impulsive move causing a Bullish BOS.
        Bearish OB: The last up-close candle before an impulsive move causing a Bearish BOS.
        """
        df = df.copy()

        df['OB_Bull_Price'] = np.nan
        df['OB_Bear_Price'] = np.nan
        df['OB_Volume'] = np.nan

        # We need to iterate over BOS events to trace back the origin candle.
        # For production speed, we do a localized backward loop on valid triggers.
        bos_bull_indices = df[df['BOS_Bull'] | df['CHoCH_Bull']].index
        bos_bear_indices = df[df['BOS_Bear'] | df['CHoCH_Bear']].index

        for idx in bos_bull_indices:
            # Look back up to 15 candles for the last down candle (Close < Open)
            if idx < 15: continue
            lookback = df.iloc[idx - 15:idx]
            down_candles = lookback[lookback['Close'] < lookback['Open']]
            if not down_candles.empty:
                ob_idx = down_candles.index[-1]
                df.at[idx, 'OB_Bull_Price'] = df.at[ob_idx, 'Low']  # OB mapped to the low of that candle
                df.at[idx, 'OB_Volume'] = df.at[ob_idx, 'Volume']

        for idx in bos_bear_indices:
            # Look back up to 15 candles for the last up candle (Close > Open)
            if idx < 15: continue
            lookback = df.iloc[idx - 15:idx]
            up_candles = lookback[lookback['Close'] > lookback['Open']]
            if not up_candles.empty:
                ob_idx = up_candles.index[-1]
                df.at[idx, 'OB_Bear_Price'] = df.at[ob_idx, 'High']  # OB mapped to the high of that candle
                df.at[idx, 'OB_Volume'] = df.at[ob_idx, 'Volume']

        # Forward fill the active Order Block zones
        df['Active_OB_Bull'] = df['OB_Bull_Price'].ffill()
        df['Active_OB_Bear'] = df['OB_Bear_Price'].ffill()

        return df

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master Pipeline for Phase 2.
        Processes Phase 1 outputs to generate SMC signals.
        """
        # Ensure Phase 1 required columns exist
        required_cols = ['Last_Swing_High', 'Last_Swing_Low', 'Structure_High', 'Structure_Low']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required Phase 1 output column: {col}")

        df = self.detect_fvg(df)
        df = self.detect_bos_and_choch(df)
        df = self.detect_liquidity_sweeps(df)
        df = self.detect_order_blocks(df)

        return df


if __name__ == "__main__":
    from market_structure import MarketStructureEngine

    print("Testing Phase 2: SMC Engine...")

    # 1. Create Mock Data
    np.random.seed(10)
    prices = np.linspace(100, 150, 50).tolist() + np.linspace(150, 120, 30).tolist()
    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 3 for p in prices],
        'Low': [p - 3 for p in prices],
        'Close': [p + np.random.normal(0, 1) for p in prices],
        'Volume': np.random.randint(500, 2000, 80)
    })

    # 2. Run Phase 1
    mse = MarketStructureEngine(swing_window=3)
    # Use private method to extract the dataframe directly for testing purposes
    df_structured = mse._classify_market_structure(mse._detect_pivots(df))

    # 3. Run Phase 2
    smc = SMCEngine()
    df_smc = smc.analyze(df_structured)

    # Show subset of relevant columns
    output_cols = ['Close', 'FVG_Bull', 'BOS_Bull', 'CHoCH_Bear', 'Sweep_Bull', 'Active_OB_Bull']
    print(df_smc[output_cols].tail(15))