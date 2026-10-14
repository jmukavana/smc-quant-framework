import pandas as pd
import numpy as np


class SetupEngine:
    """
    Phase 4: Institutional Trade Setups Engine
    Scans enriched market data for 7 high-probability SMC/Volume confluences.
    """

    @staticmethod
    def setup_1_sweep_bos_ob(df: pd.DataFrame) -> pd.DataFrame:
        """
        SETUP 1: Liquidity Sweep + BOS + Order Block Retest
        Logic: Smart money sweeps external liquidity (stops), rapidly displaces price
        causing a Break of Structure, and we enter on the retracement to the origin Order Block.
        """
        # Look back 20 periods for a recent Sweep followed by a BOS
        recent_sweep_bull = df['Sweep_Bull'].rolling(20).max() > 0
        recent_bos_bull = df['BOS_Bull'].rolling(15).max() > 0

        # Current condition: Price is retesting the Active Bullish Order Block
        ob_retest_bull = (df['Low'] <= df['Active_OB_Bull']) & (df['Close'] > df['Active_OB_Bull'])

        df['Setup_1_Long'] = recent_sweep_bull & recent_bos_bull & ob_retest_bull

        # Bearish equivalent
        recent_sweep_bear = df['Sweep_Bear'].rolling(20).max() > 0
        recent_bos_bear = df['BOS_Bear'].rolling(15).max() > 0
        ob_retest_bear = (df['High'] >= df['Active_OB_Bear']) & (df['Close'] < df['Active_OB_Bear'])

        df['Setup_1_Short'] = recent_sweep_bear & recent_bos_bear & ob_retest_bear

        return df

    @staticmethod
    def setup_2_choch_volume(df: pd.DataFrame) -> pd.DataFrame:
        """
        SETUP 2: Change of Character (CHoCH) + Volume Confirmation
        Logic: A trend reversal is only valid if backed by institutional volume displacement.
        """
        # CHoCH must be present, and the Volume Displacement Score must be high (> 60)
        df['Setup_2_Long'] = df['CHoCH_Bull'] & (df['CHoCH_Strength'] > 60)
        df['Setup_2_Short'] = df['CHoCH_Bear'] & (df['CHoCH_Strength'] > 60)

        return df

    @staticmethod
    def setup_3_fvg_volume_cluster(df: pd.DataFrame, poc_price: float) -> pd.DataFrame:
        """
        SETUP 3: Fair Value Gap Retest + Volume Cluster (POC Proximity)
        Logic: FVGs are magnets. If an FVG aligns closely with the Volume Point of Control,
        it is a highly defensible area for institutions.
        """
        # Price dips into the FVG
        fvg_retest_bull = (df['Low'] <= df['High'].shift(2)) & df['FVG_Bull'].rolling(10).max() > 0

        # FVG happens to be near the Point of Control (within a 0.5% tolerance)
        near_poc = abs(df['Close'] - poc_price) / poc_price < 0.005

        df['Setup_3_Long'] = fvg_retest_bull & near_poc

        # Bearish equivalent
        fvg_retest_bear = (df['High'] >= df['Low'].shift(2)) & df['FVG_Bear'].rolling(10).max() > 0
        df['Setup_3_Short'] = fvg_retest_bear & near_poc

        return df

    @staticmethod
    def setup_4_accumulation_breakout(df: pd.DataFrame, vah: float, val: float) -> pd.DataFrame:
        """
        SETUP 4: Accumulation Breakout
        Logic: Price was in a tight accumulation squeeze (Phase 3), and suddenly
        breaks out of the Value Area with high volume.
        """
        was_accumulating = df['Is_Accumulating'].rolling(5).max() > 0

        # Breakout above VAH
        bull_breakout = (df['Close'] > vah) & (df['Close'].shift(1) <= vah)
        high_vol = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5

        df['Setup_4_Long'] = was_accumulating & bull_breakout & high_vol

        # Breakdown below VAL (Distribution breakdown)
        bear_breakout = (df['Close'] < val) & (df['Close'].shift(1) >= val)
        df['Setup_4_Short'] = was_accumulating & bear_breakout & high_vol

        return df

    @staticmethod
    def setup_5_sr_flip(df: pd.DataFrame) -> pd.DataFrame:
        """
        SETUP 5: Support/Resistance Flip (Mitigation Block)
        Logic: A broken Swing High becomes support. Price returns to mitigate trapped shorts.
        """
        # Bullish S/R Flip: We broke above a Swing High previously. Now we retest it.
        broken_resistance = df['Last_Swing_High']
        retest_support = (df['Low'] <= broken_resistance) & (df['Close'] > broken_resistance)
        is_bull_trend = df['Structure_High'] == 'HH'

        df['Setup_5_Long'] = retest_support & is_bull_trend

        # Bearish S/R Flip
        broken_support = df['Last_Swing_Low']
        retest_resistance = (df['High'] >= broken_support) & (df['Close'] < broken_support)
        is_bear_trend = df['Structure_Low'] == 'LL'

        df['Setup_5_Short'] = retest_resistance & is_bear_trend

        return df

    @staticmethod
    def setup_6_open_drive(df: pd.DataFrame) -> pd.DataFrame:
        """
        SETUP 6: Open Drive Strategy
        Logic: Usually executed at London or NY open. Price leaves the opening range
        aggressively. Measured by massive volume and a candle closing near its extreme.
        """
        # Ensure 'Datetime' index or column exists for session timing in live environment.
        # Here we approximate via volatility expansion and closing strength.
        closing_strength_bull = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8) > 0.85
        closing_strength_bear = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-8) > 0.85

        vol_surge = df['Volume'] > df['Volume'].rolling(50).mean() * 2.5

        df['Setup_6_Long'] = vol_surge & closing_strength_bull
        df['Setup_6_Short'] = vol_surge & closing_strength_bear

        return df

    @staticmethod
    def setup_7_trend_continuation(df: pd.DataFrame) -> pd.DataFrame:
        """
        SETUP 7: Premium/Discount Trend Continuation
        Logic: In a strong trend, buy pullbacks into the "Discount Zone" (< 50% of the last swing).
        """
        swing_range = df['Last_Swing_High'] - df['Last_Swing_Low']
        equilibrium = df['Last_Swing_Low'] + (swing_range * 0.5)

        # Bullish: Trend is HH/HL, Price pulls back below Equilibrium (Discount), Rejects
        in_discount = df['Low'] < equilibrium
        bull_rejection = df['Close'] > df['Open']  # Green candle in discount
        df['Setup_7_Long'] = (df['Structure_High'] == 'HH') & in_discount & bull_rejection

        # Bearish: Trend is LH/LL, Price pulls up above Equilibrium (Premium), Rejects
        in_premium = df['High'] > equilibrium
        bear_rejection = df['Close'] < df['Open']  # Red candle in premium
        df['Setup_7_Short'] = (df['Structure_Low'] == 'LL') & in_premium & bear_rejection

        return df

    def scan_all_setups(self, df: pd.DataFrame, vp_metrics: dict) -> pd.DataFrame:
        """
        Master Pipeline for Phase 4.
        Executes all 7 setup scans and creates an aggregated 'Action' column.
        """
        df = self.setup_1_sweep_bos_ob(df)
        df = self.setup_2_choch_volume(df)
        df = self.setup_3_fvg_volume_cluster(df, poc_price=vp_metrics['POC'])

        # Ensure Is_Accumulating exists from Phase 3, else mock it for safety
        if 'Is_Accumulating' not in df.columns:
            df['Is_Accumulating'] = False

        df = self.setup_4_accumulation_breakout(df, vah=vp_metrics['VAH'], val=vp_metrics['VAL'])
        df = self.setup_5_sr_flip(df)
        df = self.setup_6_open_drive(df)
        df = self.setup_7_trend_continuation(df)

        # Aggregate signals for easier downstream processing
        long_cols = [c for c in df.columns if 'Setup_' in c and 'Long' in c]
        short_cols = [c for c in df.columns if 'Setup_' in c and 'Short' in c]

        df['Signal_Long'] = df[long_cols].any(axis=1)
        df['Signal_Short'] = df[short_cols].any(axis=1)

        # Collect exactly which setups fired for the AI Market Analyst and Trade Journal
        df['Active_Setups'] = df.apply(
            lambda row: [col for col in long_cols + short_cols if row[col] == True], axis=1
        )

        return df