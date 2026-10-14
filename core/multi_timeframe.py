import pandas as pd
import numpy as np
from typing import Dict, List


class MultiTimeframeAnalyzer:
    """
    Phase 5: Multi-Timeframe Alignment Engine
    Ingests multiple enriched timeframes to construct a composite institutional bias.
    Prevents execution of lower-timeframe setups against higher-timeframe macro trends.
    """

    def __init__(self, htf_weights: Dict[str, float] = None):
        """
        :param htf_weights: Dictionary defining the weight of each timeframe for scoring.
        """
        # Default institutional weighting (Weekly/Daily dictate macro, 4H/1H dictate intermediate)
        self.htf_weights = htf_weights or {
            '1W': 1.0,
            '1D': 0.8,
            '4H': 0.6,
            '1H': 0.4,
            '15M': 0.2
        }

    def _extract_tf_trend(self, df: pd.DataFrame) -> int:
        """
        Extracts the structural trend from a Phase-1 enriched DataFrame.
        Returns: +1 for Bullish, -1 for Bearish, 0 for Sideways/Neutral.
        """
        # Using the classification from Phase 1 Market Structure Engine
        recent_high_struct = df['Structure_High'].iloc[-1]
        recent_low_struct = df['Structure_Low'].iloc[-1]

        if recent_high_struct == 'HH' and recent_low_struct == 'HL':
            return 1  # Strong Bullish
        elif recent_high_struct == 'LH' and recent_low_struct == 'LL':
            return -1  # Strong Bearish
        elif recent_high_struct == 'HH' and recent_low_struct == 'LL':
            return 0  # Expanding (Neutral/Volatile)
        elif recent_high_struct == 'LH' and recent_low_struct == 'HL':
            return 0  # Contracting (Neutral/Consolidation)

        return 0

    def calculate_composite_bias(self, tf_data: Dict[str, pd.DataFrame]) -> dict:
        """
        Calculates a weighted macro bias score.
        :param tf_data: Dictionary of DataFrames keyed by timeframe (e.g., {'1D': df_1d, '4H': df_4h})
        """
        composite_score = 0.0
        max_possible_score = 0.0
        tf_states = {}

        for tf, df in tf_data.items():
            if tf not in self.htf_weights:
                continue

            weight = self.htf_weights[tf]
            trend_val = self._extract_tf_trend(df)

            composite_score += trend_val * weight
            max_possible_score += weight

            tf_states[tf] = {
                "trend": "Bullish" if trend_val == 1 else "Bearish" if trend_val == -1 else "Neutral",
                "last_swing_high": float(df['Last_Swing_High'].iloc[-1]),
                "last_swing_low": float(df['Last_Swing_Low'].iloc[-1])
            }

        # Normalize score between -100 and +100
        normalized_bias = (composite_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        if normalized_bias >= 40:
            bias_direction = "Bullish"
        elif normalized_bias <= -40:
            bias_direction = "Bearish"
        else:
            bias_direction = "Neutral"

        return {
            "bias_score": normalized_bias,
            "bias_direction": bias_direction,
            "timeframe_states": tf_states
        }

    def filter_ltf_setups(self,
                          ltf_df: pd.DataFrame,
                          composite_bias: dict,
                          strict_alignment: bool = True) -> pd.DataFrame:
        """
        Applies the Higher Timeframe lock to the Lower Timeframe (Entry) setups.
        Blocks long setups in a macro bear market, and short setups in a macro bull market.
        """
        df = ltf_df.copy()

        # Ensure Phase 4 columns exist
        if 'Signal_Long' not in df.columns or 'Signal_Short' not in df.columns:
            return df

        macro_direction = composite_bias['bias_direction']

        df['MTF_Aligned_Long'] = False
        df['MTF_Aligned_Short'] = False

        if strict_alignment:
            # Only allow trades strictly in the direction of the HTF Bias
            if macro_direction == "Bullish":
                df['MTF_Aligned_Long'] = df['Signal_Long']
                # Signal_Short remains False
            elif macro_direction == "Bearish":
                df['MTF_Aligned_Short'] = df['Signal_Short']
                # Signal_Long remains False
            else:
                # In Neutral macro, only allow Mean Reversion (Accumulation Breakout / Sweeps)
                # We do not take pure trend continuation setups in sideways chop.
                df['MTF_Aligned_Long'] = df['Signal_Long'] & (df['Is_Accumulating'] | df['Sweep_Bull'])
                df['MTF_Aligned_Short'] = df['Signal_Short'] & (df['Is_Accumulating'] | df['Sweep_Bear'])
        else:
            # Lenient alignment: Allows counter-trend trades but marks them for reduced position sizing
            df['MTF_Aligned_Long'] = df['Signal_Long']
            df['MTF_Aligned_Short'] = df['Signal_Short']
            df['Counter_Trend_Long'] = df['Signal_Long'] & (macro_direction == "Bearish")
            df['Counter_Trend_Short'] = df['Signal_Short'] & (macro_direction == "Bullish")

        return df

    def check_htf_poi_collision(self, ltf_price: float, tf_states: dict, tolerance_pct: float = 0.002) -> dict:
        """
        Checks if the current Lower Timeframe price is colliding with a
        Higher Timeframe Point of Interest (Swing High/Low).
        This is used to target exits (Take Profit) or anticipate HTF rejections.
        """
        collisions = []
        for tf, state in tf_states.items():
            sh = state['last_swing_high']
            sl = state['last_swing_low']

            # Check proximity to HTF Swing High (Resistance)
            if abs(ltf_price - sh) / sh <= tolerance_pct:
                collisions.append(f"{tf} Resistance ({sh})")

            # Check proximity to HTF Swing Low (Support)
            if abs(ltf_price - sl) / sl <= tolerance_pct:
                collisions.append(f"{tf} Support ({sl})")

        return {
            "has_collision": len(collisions) > 0,
            "poi_collisions": collisions
        }


if __name__ == "__main__":
    print("Testing Phase 5: Multi-Timeframe Engine...")


    # Mock Phase 1 output for different timeframes
    def create_mock_tf_data(high_struct, low_struct, price):
        return pd.DataFrame({
            'Structure_High': [high_struct],
            'Structure_Low': [low_struct],
            'Last_Swing_High': [price * 1.05],
            'Last_Swing_Low': [price * 0.95],
            'Close': [price]
        })


    tf_data = {
        '1D': create_mock_tf_data('HH', 'HL', 40000),  # Bullish
        '4H': create_mock_tf_data('HH', 'HL', 40000),  # Bullish
        '1H': create_mock_tf_data('LH', 'HL', 40000)  # Neutral/Consolidating
    }

    mtf_engine = MultiTimeframeAnalyzer()
    bias = mtf_engine.calculate_composite_bias(tf_data)

    import json

    print("\n--- Composite MTF Bias ---")
    print(json.dumps(bias, indent=4))

    # Check Collision Example
    current_ltf_price = 41980  # Pushing near the 1D/4H Swing High (40000 * 1.05 = 42000)
    collisions = mtf_engine.check_htf_poi_collision(current_ltf_price, bias['timeframe_states'])

    print("\n--- HTF Collision Check ---")
    print(f"Current Price: {current_ltf_price}")
    print(json.dumps(collisions, indent=4))