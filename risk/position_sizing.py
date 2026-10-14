from risk.drawdown_manager import DrawdownManager

class PositionSizer:
    """
    Phase 6B: Dynamic Position Sizing & Target Calculation Engine
    Calculates exact lot size based on structural Stop Loss distance.
    Enforces minimum 1:3 RR rule.
    """

    def __init__(self, drawdown_manager: DrawdownManager):
        self.dd_manager = drawdown_manager

    def determine_risk_tier(self, setup_confidence: str) -> float:
        """
        Allocates risk based on AI/ML Setup Confidence.
        High Confidence = 1.0%, Standard = 0.5%, Low/Counter-Trend = 0.25%
        """
        tiers = {
            "HIGH": 0.0100,  # 1.00%
            "MEDIUM": 0.0050,  # 0.50%
            "LOW": 0.0025  # 0.25%
        }
        return tiers.get(setup_confidence.upper(), 0.0025)

    def calculate_trade_parameters(self,
                                   entry_price: float,
                                   sl_price: float,
                                   setup_confidence: str = "MEDIUM",
                                   asset_multiplier: float = 1.0) -> dict:
        """
        Calculates position size, exact dollar risk, and validates Target (TP).
        :param asset_multiplier: E.g., 100,000 for standard Forex lot, 1 for Crypto.
        """
        # 1. Check Circuit Breakers
        risk_status = self.dd_manager.can_trade()
        if not risk_status['can_trade']:
            return {"valid": False, "reason": risk_status['reason']}

        # 2. Prevent invalid SL placements (e.g., SL == Entry)
        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            return {"valid": False, "reason": "Stop Loss distance cannot be zero."}

        # 3. Determine Risk % and Cash Amount
        risk_pct = self.determine_risk_tier(setup_confidence)
        account_balance = self.dd_manager.current_balance
        cash_risk = account_balance * risk_pct

        # 4. Calculate Position Size
        # Size = (Risk Amount) / (Distance to SL * Multiplier)
        risk_per_unit = sl_distance * asset_multiplier
        position_size = cash_risk / risk_per_unit

        # 5. Calculate Minimum 1:3 Target (Take Profit)
        trade_direction = "LONG" if entry_price > sl_price else "SHORT"
        min_reward_distance = sl_distance * 3
        preferred_reward_distance = sl_distance * 5

        if trade_direction == "LONG":
            min_tp = entry_price + min_reward_distance
            pref_tp = entry_price + preferred_reward_distance
        else:
            min_tp = entry_price - min_reward_distance
            pref_tp = entry_price - preferred_reward_distance

        return {
            "valid": True,
            "direction": trade_direction,
            "entry": entry_price,
            "stop_loss": sl_price,
            "take_profit_min_3R": min_tp,
            "take_profit_pref_5R": pref_tp,
            "position_size": position_size,
            "risk_amount": cash_risk,
            "risk_pct": risk_pct * 100,
            "account_balance": account_balance
        }


if __name__ == "__main__":
    from drawdown_manager import DrawdownManager

    print("Testing Phase 6: Risk Management Engine...")

    # Initialize with $100,000 account
    dd_manager = DrawdownManager(initial_balance=100000.0)
    sizer = PositionSizer(dd_manager)

    # Mock Trade Setup:
    # AI Analyst gave "HIGH" confidence for a Long on BTC.
    # Entry at 65000, Structure (Swing Low) at 64000.
    trade_params = sizer.calculate_trade_parameters(
        entry_price=65000.0,
        sl_price=64000.0,
        setup_confidence="HIGH",
        asset_multiplier=1.0  # Crypto is 1:1. Forex might be 100000.
    )

    import json

    print("\n--- Trade Calculation (100k Account, HIGH Confidence) ---")
    print(json.dumps(trade_params, indent=4))

    # Simulate a terrible day: 4 back-to-back losses (-3.5% DD)
    print("\n--- Simulating Drawdown Circuit Breaker ---")
    dd_manager.update_balance(-3500.0)  # -3.5% loss on day 1

    # Next trade attempt
    blocked_trade = sizer.calculate_trade_parameters(
        entry_price=65000.0,
        sl_price=64000.0,
        setup_confidence="HIGH"
    )
    print(json.dumps(blocked_trade, indent=4))