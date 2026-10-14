import json
import os
from typing import Dict, Any

# We use the OpenAI library. This can easily be swapped for
# an open-source local model like LLaMA-3 via Ollama/vLLM for privacy.
try:
    import openai
except ImportError:
    openai = None


class AIMarketAnalyst:
    """
    Phase 7: Generative AI Market Analyst
    Ingests technical states across all engines and uses an LLM to generate
    a synthesized market narrative, anomaly detection, and a parsed JSON execution signal.
    """

    def __init__(self, model: str = "gpt-4-turbo-preview", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "mock_key")

        if openai and self.api_key != "mock_key":
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def _construct_prompt(self,
                          market_structure: Dict[str, Any],
                          smc_signals: Dict[str, Any],
                          volume_profile: Dict[str, Any],
                          mtf_bias: Dict[str, Any],
                          news_context: str) -> str:
        """
        Translates raw algorithmic data dictionaries into a structured text prompt.
        """
        prompt = f"""
        You are a Senior Institutional Quantitative Analyst and Algorithmic Trader.
        Analyze the current market state and rank the best trading opportunity.

        --- CURRENT MARKET STATE ---
        1. MACRO BIAS (Phase 5): {mtf_bias.get('bias_direction')} (Score: {mtf_bias.get('bias_score')})

        2. MARKET STRUCTURE (Phase 1):
        - Trend: {market_structure.get('trend_direction')} ({market_structure.get('trend_strength')})
        - Current Wave: {market_structure.get('current_wave')}

        3. SMART MONEY CONCEPTS (Phase 2 & 4):
        - Active Setups Fired: {smc_signals.get('Active_Setups', 'None')}
        - Order Block Context: {smc_signals.get('OB_Context', 'Neutral')}

        4. VOLUME PROFILE (Phase 3):
        - Accumulation Phase: {volume_profile.get('Is_Accumulating')}
        - Failed Auctions Detected: High={volume_profile.get('Failed_Auction_High')}, Low={volume_profile.get('Failed_Auction_Low')}

        5. MACRO / NEWS CONTEXT (Phase 9):
        - {news_context}

        --- INSTRUCTIONS ---
        Based on the data above:
        1. Explain the current market regime.
        2. Detect any anomalies (e.g., Bullish SMC setup in a Bearish Macro MTF).
        3. Determine the ultimate trade direction (LONG, SHORT, or NEUTRAL).
        4. Provide a Confidence Score (0-100).
        5. Provide a Probability of Success estimate (0.00 - 1.00).
        6. Determine Risk Tier (HIGH, MEDIUM, LOW).

        You MUST respond ONLY in valid JSON format matching the exact schema below. Do not include markdown formatting or extra text.

        {{
            "trade_direction": "LONG|SHORT|NEUTRAL",
            "confidence": 85,
            "probability": 0.75,
            "reasoning": "Brief 2-sentence institutional explanation...",
            "anomaly_detected": "Describe any conflicting data or None",
            "risk_score": "MEDIUM"
        }}
        """
        return prompt

    def analyze_market_context(self,
                               market_structure: Dict[str, Any],
                               smc_signals: Dict[str, Any],
                               volume_profile: Dict[str, Any],
                               mtf_bias: Dict[str, Any],
                               news_context: str = "No imminent high-impact news.") -> Dict[str, Any]:
        """
        Calls the LLM to process the market data.
        Returns a strongly typed JSON dictionary for the execution engine.
        """
        prompt = self._construct_prompt(market_structure, smc_signals, volume_profile, mtf_bias, news_context)

        # Mock response for testing without an API key
        if not self.client or self.api_key == "mock_key":
            print("[WARN] OpenAI API key not found. Returning synthesized mock analysis.")
            return {
                "trade_direction": "LONG" if "Setup_1_Long" in smc_signals.get('Active_Setups', []) else "NEUTRAL",
                "confidence": 82,
                "probability": 0.78,
                "reasoning": "[MOCK] Structural alignment with Daily MTF. Accumulation breakout detected off the POC.",
                "anomaly_detected": "None",
                "risk_score": "MEDIUM"
            }

        try:
            # Production LLM Call enforcing JSON response format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a precise institutional trading algorithm. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},  # GPT-4/3.5 Turbo JSON Mode
                temperature=0.2,  # Low temperature for analytical consistency
                max_tokens=300
            )

            result_str = response.choices[0].message.content
            parsed_result = json.loads(result_str)
            return parsed_result

        except json.JSONDecodeError as e:
            print(f"[ERROR] LLM did not return valid JSON: {e}")
            return {"trade_direction": "NEUTRAL", "confidence": 0, "reasoning": "JSON parse error."}
        except Exception as e:
            print(f"[ERROR] LLM API Call Failed: {e}")
            return {"trade_direction": "NEUTRAL", "confidence": 0, "reasoning": "API connection error."}


if __name__ == "__main__":
    print("Testing Phase 7: AI Market Analyst...")

    # Simulating data generated from Phases 1-5
    mock_market_structure = {
        "trend_direction": "Bullish",
        "trend_strength": "Strong",
        "current_wave": "Corrective"
    }

    mock_smc_signals = {
        "Active_Setups": ["Setup_1_Long", "Setup_3_Long"],
        "OB_Context": "Retesting Daily Bullish Order Block"
    }

    mock_volume_profile = {
        "Is_Accumulating": True,
        "Failed_Auction_High": False,
        "Failed_Auction_Low": True
    }

    mock_mtf_bias = {
        "bias_direction": "Bullish",
        "bias_score": 85.0
    }

    mock_news = "NFP (Non-Farm Payroll) release in 4 hours."

    ai = AIMarketAnalyst()
    analysis = ai.analyze_market_context(
        mock_market_structure,
        mock_smc_signals,
        mock_volume_profile,
        mock_mtf_bias,
        mock_news
    )

    print("\n--- AI Analyst Output ---")
    print(json.dumps(analysis, indent=4))