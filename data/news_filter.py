import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import time


class NewsFilter:
    """
    Phase 9: Institutional News & Volatility Filter
    Fetches the economic calendar and halts trading +/- 30 minutes
    around High Impact (Red Folder) events to prevent catastrophic slippage.
    """

    def __init__(self, buffer_minutes: int = 30):
        """
        :param buffer_minutes: Minutes before and after the event to halt trading.
        """
        self.buffer_minutes = buffer_minutes
        self.news_events = []
        self.last_fetch_time = None
        self.feed_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

    def fetch_calendar(self):
        """
        Fetches and parses the ForexFactory weekly XML calendar.
        Standard in retail/quant algo trading as it requires no API key.
        """
        try:
            req = urllib.request.Request(self.feed_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            events = []

            for event in root.findall('event'):
                # Extract event details
                impact = event.find('impact').text.strip()
                if impact != 'High':
                    continue  # Only track Red Folder / High Impact news

                country = event.find('country').text.strip()
                date_str = event.find('date').text.strip()
                time_str = event.find('time').text.strip()
                title = event.find('title').text.strip()

                # Handle "All Day" or undefined times
                if time_str == 'All Day' or not time_str:
                    continue

                # Parse datetime (FF feed is typically EST/EDT by default, but let's parse
                # carefully. The dates are usually formatted as '10-25-2023' and '8:30am')
                try:
                    # Construct full datetime string
                    dt_str = f"{date_str} {time_str}"
                    event_dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")

                    # Convert to UTC (FF default base is typically US Eastern Time, we assume EST for safety here,
                    # though production systems should query an API with explicit UTC timestamps).
                    # For absolute precision, production systems use TradingEconomics API or Bloomberg.
                    # We will treat the parsed time as an offset-naive time and assign it an approximation for now.
                    event_dt = event_dt + timedelta(hours=5)  # Approx EST to UTC conversion
                    event_dt = event_dt.replace(tzinfo=timezone.utc)

                    events.append({
                        "currency": country,
                        "title": title,
                        "time": event_dt
                    })
                except Exception as parse_e:
                    continue

            self.news_events = events
            self.last_fetch_time = datetime.now(timezone.utc)
            print(f"[NewsFilter] Successfully loaded {len(self.news_events)} High Impact events.")

        except Exception as e:
            print(f"[ERROR] Failed to fetch economic calendar: {e}")
            # If API fails, we fail-safe (do not block forever, but log the error)

    def is_trading_allowed(self, current_time: datetime, symbol: str) -> dict:
        """
        Checks if the current time is inside the exclusion window of a High Impact event.
        :param current_time: The current UTC datetime.
        :param symbol: The asset being traded (e.g., 'EURUSD', 'BTCUSDT').
        """
        # Refresh data if it's older than 12 hours
        if self.last_fetch_time is None or (current_time - self.last_fetch_time).total_seconds() > 43200:
            self.fetch_calendar()

        if not self.news_events:
            return {"allowed": True, "reason": "No high impact news loaded."}

        # Extract base and quote currencies from symbol (e.g., EURUSD -> EUR, USD)
        # For Crypto like BTCUSDT, we usually map USDT to USD.
        currencies_to_check = []
        if len(symbol) >= 6:
            currencies_to_check.append(symbol[:3])
            currencies_to_check.append(symbol[3:6].replace('USDT', 'USD'))
        else:
            currencies_to_check.append(symbol)  # Fallback

        # USD events heavily impact Crypto and Indices as well
        if "BTC" in symbol or "ETH" in symbol or "SPX" in symbol:
            if "USD" not in currencies_to_check:
                currencies_to_check.append("USD")

        for event in self.news_events:
            # Check if event currency matters for our traded symbol
            if event['currency'] not in currencies_to_check:
                continue

            time_diff = (current_time - event['time']).total_seconds() / 60.0  # difference in minutes

            # Check if we are within the buffer window (-30 to +30)
            if -self.buffer_minutes <= time_diff <= self.buffer_minutes:
                return {
                    "allowed": False,
                    "reason": f"News Block: '{event['title']}' for {event['currency']} at {event['time'].strftime('%H:%M UTC')}. Time diff: {int(time_diff)} mins."
                }

        return {"allowed": True, "reason": "Clear of high impact news."}


if __name__ == "__main__":
    print("Testing Phase 9: News Filter...")

    nf = NewsFilter(buffer_minutes=30)
    nf.fetch_calendar()

    # Test 1: Assume it is right now
    now_utc = datetime.now(timezone.utc)
    status_eurusd = nf.is_trading_allowed(now_utc, "EURUSD")
    status_btcusdt = nf.is_trading_allowed(now_utc, "BTCUSDT")

    print("\n--- Current News Status ---")
    print(f"EUR/USD Status: {status_eurusd}")
    print(f"BTC/USDT Status: {status_btcusdt}")

    # Test 2: Simulating a collision
    if nf.news_events:
        # Take the first event and pretend current time is 10 minutes before it
        mock_event_time = nf.news_events[0]['time']
        mock_current_time = mock_event_time - timedelta(minutes=10)
        test_currency = nf.news_events[0]['currency']

        # Test symbol (e.g., if currency is USD, we test GBPUSD)
        test_symbol = f"GBP{test_currency}" if test_currency == "USD" else f"{test_currency}USD"

        mock_status = nf.is_trading_allowed(mock_current_time, test_symbol)
        print(f"\n--- Simulated Collision Status for {test_symbol} ---")
        print(f"Event: {nf.news_events[0]['title']} at {mock_event_time}")
        print(f"Simulated Current Time: {mock_current_time}")
        print(f"Result: {mock_status}")