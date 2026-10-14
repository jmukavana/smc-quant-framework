import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class VolumeProfileEngine:
    """
    Phase 3: Volume Profile & Institutional Accumulation Engine
    Calculates POC, VAH, VAL, Volume Nodes, and detects Accumulation/Distribution.
    """

    def __init__(self, n_bins: int = 50, value_area_pct: float = 0.70):
        """
        :param n_bins: Number of price levels (bins) to divide the profile into.
        :param value_area_pct: Percentage of total volume that constitutes the Value Area (usually 70%).
        """
        self.n_bins = n_bins
        self.va_pct = value_area_pct

    def _build_profile(self, df: pd.DataFrame) -> dict:
        """
        Constructs the Volume Profile by distributing candle volume across price bins.
        Returns the Profile distribution, POC, VAH, and VAL.
        """
        min_price = df['Low'].min()
        max_price = df['High'].max()

        # Create price bins
        bins = np.linspace(min_price, max_price, self.n_bins)
        profile = np.zeros(self.n_bins - 1)

        # Vectorized volume distribution
        # For OHLCV data, a robust approximation is to distribute volume evenly
        # between the Low and High of each candle.
        highs = df['High'].values
        lows = df['Low'].values
        vols = df['Volume'].values

        for i in range(len(df)):
            if highs[i] == lows[i]:
                # Avoid division by zero on zero-range candles
                idx = np.digitize(highs[i], bins) - 1
                idx = min(max(idx, 0), len(profile) - 1)
                profile[idx] += vols[i]
            else:
                # Find which bins this candle intersects
                candle_bins = np.where((bins[:-1] <= highs[i]) & (bins[1:] >= lows[i]))[0]
                if len(candle_bins) > 0:
                    vol_per_bin = vols[i] / len(candle_bins)
                    profile[candle_bins] += vol_per_bin

        # Identify Point of Control (POC)
        poc_idx = np.argmax(profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Calculate Value Area (VAH & VAL) via continuous outward expansion from POC
        total_vol = np.sum(profile)
        target_vol = total_vol * self.va_pct

        current_vol = profile[poc_idx]
        up_idx = poc_idx
        down_idx = poc_idx

        while current_vol < target_vol and (up_idx < len(profile) - 1 or down_idx > 0):
            vol_up = profile[up_idx + 1] if up_idx < len(profile) - 1 else -1
            vol_down = profile[down_idx - 1] if down_idx > 0 else -1

            if vol_up >= vol_down and vol_up != -1:
                up_idx += 1
                current_vol += profile[up_idx]
            elif vol_down > vol_up and vol_down != -1:
                down_idx -= 1
                current_vol += profile[down_idx]
            else:
                break

        vah_price = bins[up_idx + 1]
        val_price = bins[down_idx]

        return {
            "bins": bins,
            "profile": profile,
            "POC": poc_price,
            "VAH": vah_price,
            "VAL": val_price,
            "Total_Volume": total_vol
        }

    def _find_volume_nodes(self, profile_data: dict) -> dict:
        """
        Identifies High Volume Nodes (HVN) and Low Volume Nodes (LVN).
        HVNs act as gravity (prices revert here). LVNs act as rejection zones.
        """
        profile = profile_data['profile']
        bins = profile_data['bins']

        # High Volume Nodes (Peaks in the profile)
        hvn_indices, _ = find_peaks(profile, distance=3, prominence=np.mean(profile) * 0.2)
        hvn_prices = [(bins[i] + bins[i + 1]) / 2 for i in hvn_indices]

        # Low Volume Nodes (Troughs in the profile) -> invert the profile to find peaks of the inversion
        inverted_profile = np.max(profile) - profile
        lvn_indices, _ = find_peaks(inverted_profile, distance=3, prominence=np.mean(profile) * 0.2)
        lvn_prices = [(bins[i] + bins[i + 1]) / 2 for i in lvn_indices]

        return {"HVNs": hvn_prices, "LVNs": lvn_prices}

    def _detect_accumulation_distribution(self, df: pd.DataFrame, profile_data: dict) -> pd.DataFrame:
        """
        Detects Institutional Accumulation and Distribution.
        Rules:
        - Price constrained within Value Area (VAL to VAH).
        - Low volatility (ATR squeezing).
        - Volume profile shows a massive bell curve at POC.
        """
        df = df.copy()

        # Calculate recent volatility (ATR proxy)
        df['TR'] = df[['High']].max(axis=1) - df[['Low']].min(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()

        poc = profile_data['POC']
        vah = profile_data['VAH']
        val = profile_data['VAL']

        # Institutional Accumulation typically happens in a tight range near the POC
        # where the close is consistently between VAL and VAH, and volatility is dropping.
        df['In_Value_Area'] = (df['Close'] <= vah) & (df['Close'] >= val)
        df['Near_POC'] = abs(df['Close'] - poc) < (df['ATR'] * 0.5)

        # Squeeze detection: Current ATR is significantly less than historical ATR
        df['ATR_Squeeze'] = df['ATR'] < df['ATR'].rolling(50).mean() * 0.8

        # Accumulation State: In VA + Squeezing ATR + Hovering near POC for at least 10 periods
        df['Accumulation_Phase'] = df['In_Value_Area'] & df['ATR_Squeeze'] & df['Near_POC']
        df['Is_Accumulating'] = df['Accumulation_Phase'].rolling(10).sum() >= 7  # 7 out of last 10 candles

        # Rejection Zones (Failed Auctions)
        # Price wicks heavily outside VAH/VAL into an LVN, but closes back inside the VA.
        df['Failed_Auction_High'] = (df['High'] > vah) & (df['Close'] < vah)
        df['Failed_Auction_Low'] = (df['Low'] < val) & (df['Close'] > val)

        return df

    def analyze(self, df: pd.DataFrame, lookback: int = 100) -> dict:
        """
        Master Pipeline for Phase 3.
        Creates a rolling composite volume profile for the most recent `lookback` periods.
        """
        if len(df) < lookback:
            working_df = df
        else:
            working_df = df.iloc[-lookback:]

        # 1. Build Base Profile
        profile_data = self._build_profile(working_df)

        # 2. Extract Nodes
        nodes = self._find_volume_nodes(profile_data)

        # 3. Detect Regimes (Accumulation / Rejections)
        df_enriched = self._detect_accumulation_distribution(df, profile_data)

        return {
            "POC": profile_data['POC'],
            "VAH": profile_data['VAH'],
            "VAL": profile_data['VAL'],
            "HVNs": nodes['HVNs'],
            "LVNs": nodes['LVNs'],
            "Is_Accumulating": bool(df_enriched['Is_Accumulating'].iloc[-1]),
            "Failed_Auction_High": bool(df_enriched['Failed_Auction_High'].iloc[-1]),
            "Failed_Auction_Low": bool(df_enriched['Failed_Auction_Low'].iloc[-1]),
            "Enriched_Data": df_enriched
        }


if __name__ == "__main__":
    print("Testing Phase 3: Volume Profile Engine...")

    # 1. Create Mock Data (representing a sideways range building accumulation)
    np.random.seed(42)
    base_price = 100
    # Create sideways price action with a spike
    prices = [base_price + np.random.normal(0, 0.5) for _ in range(80)]
    prices += [105, 106, 102]  # Fake breakout (Failed Auction High)
    prices += [base_price + np.random.normal(0, 0.5) for _ in range(17)]

    df = pd.DataFrame({
        'Open': prices,
        'High': [p + 1 for p in prices],
        'Low': [p - 1 for p in prices],
        'Close': [p + np.random.normal(0, 0.2) for p in prices],
        'Volume': np.random.randint(1000, 5000, 100)
    })

    # 2. Run Engine
    vp_engine = VolumeProfileEngine(n_bins=50, value_area_pct=0.70)
    vp_metrics = vp_engine.analyze(df, lookback=100)

    print("\n--- Volume Profile Analysis ---")
    print(f"Point of Control (POC): {vp_metrics['POC']:.2f}")
    print(f"Value Area High (VAH):  {vp_metrics['VAH']:.2f}")
    print(f"Value Area Low (VAL):   {vp_metrics['VAL']:.2f}")
    print(f"Is Accumulating:        {vp_metrics['Is_Accumulating']}")
    print(f"Failed Auction High:    {vp_metrics['Failed_Auction_High']}")
    print(f"High Volume Nodes:      {[round(x, 2) for x in vp_metrics['HVNs']]}")
    print(f"Low Volume Nodes:       {[round(x, 2) for x in vp_metrics['LVNs']]}")