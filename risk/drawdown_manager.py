import datetime


class DrawdownManager:
    """
    Phase 6A: Institutional Drawdown & Circuit Breaker Engine
    Tracks equity across multiple time horizons and halts trading if
    strict institutional drawdown limits are breached.
    """

    def __init__(self,
                 initial_balance: float,
                 max_daily_dd: float = 0.03,
                 max_weekly_dd: float = 0.08,
                 max_monthly_dd: float = 0.15):

        self.current_balance = initial_balance

        # Drawdown Limits
        self.limits = {
            "daily": max_daily_dd,
            "weekly": max_weekly_dd,
            "monthly": max_monthly_dd
        }

        # High watermarks (Starting balances for the period)
        self.start_balances = {
            "daily": initial_balance,
            "weekly": initial_balance,
            "monthly": initial_balance
        }

        # Time tracking
        self.last_trade_time = datetime.datetime.now(datetime.timezone.utc)

    def _rollover_periods(self, current_time: datetime.datetime):
        """
        Resets high water marks if a new day, week, or month has started.
        """
        # Daily rollover (00:00 UTC)
        if current_time.date() > self.last_trade_time.date():
            self.start_balances['daily'] = self.current_balance

        # Weekly rollover (Monday)
        if current_time.isocalendar()[1] != self.last_trade_time.isocalendar()[1]:
            self.start_balances['weekly'] = self.current_balance

        # Monthly rollover (1st of the month)
        if current_time.month != self.last_trade_time.month:
            self.start_balances['monthly'] = self.current_balance

        self.last_trade_time = current_time

    def update_balance(self, realized_pnl: float, current_time: datetime.datetime = None):
        """ Updates account equity after a closed trade. """
        if current_time is None:
            current_time = datetime.datetime.now(datetime.timezone.utc)

        self._rollover_periods(current_time)
        self.current_balance += realized_pnl

    def can_trade(self) -> dict:
        """
        Circuit Breaker check. Calculates current DD against starting balances.
        Returns a dictionary with status and reasons if halted.
        """
        drawdowns = {
            "daily": (self.start_balances['daily'] - self.current_balance) / self.start_balances['daily'],
            "weekly": (self.start_balances['weekly'] - self.current_balance) / self.start_balances['weekly'],
            "monthly": (self.start_balances['monthly'] - self.current_balance) / self.start_balances['monthly']
        }

        for period, dd in drawdowns.items():
            if dd >= self.limits[period]:
                return {
                    "can_trade": False,
                    "reason": f"CIRCUIT BREAKER TRIGGERED: {period.capitalize()} Drawdown at {dd * 100:.2f}% (Limit: {self.limits[period] * 100:.2f}%)",
                    "current_drawdowns": drawdowns
                }

        return {
            "can_trade": True,
            "reason": "Risk limits intact.",
            "current_drawdowns": drawdowns
        }