"""Binance data fetcher — live price, ATR, and historical OHLCV for CHV simulator."""

from typing import Optional, Tuple, List
import statistics
import time


def fetch_price_and_atr(
    symbol: str,
    atr_timeframe: str = "4h",
    atr_period: int = 14,
    market_type: str = 'usdm',
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Returns (current_price, atr_value, error_message).
    error_message is None on success.
    market_type: 'usdm' (default, uses fapi) or 'coinm' (uses dapi).
    """
    try:
        import requests
    except ImportError:
        return None, None, "requests library not installed. Run: pip install requests"

    symbol = symbol.upper().replace("/", "").replace("-", "")
    is_coinm = market_type.lower() == 'coinm'

    tf_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h",
        "8h": "8h", "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w",
    }
    interval = tf_map.get(atr_timeframe.lower(), "4h")

    # Fetch current price from Binance public API
    try:
        if is_coinm:
            price_url = f"https://dapi.binance.com/dapi/v1/ticker/price?symbol={symbol}"
            r = requests.get(price_url, timeout=5)
        else:
            price_url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
            r = requests.get(price_url, timeout=5)
            if r.status_code != 200:
                # Try spot
                price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                r = requests.get(price_url, timeout=5)
        data = r.json()
        # dapi returns a list when no symbol given, single dict for specific symbol
        if isinstance(data, list):
            data = data[0] if data else {}
        if "price" not in data:
            return None, None, f"Symbol {symbol} not found on Binance: {data.get('msg', 'unknown error')}"
        current_price = float(data["price"])
    except Exception as e:
        return None, None, f"Failed to fetch price: {e}"

    # Fetch OHLCV for ATR calculation
    try:
        if is_coinm:
            klines_url = (
                f"https://dapi.binance.com/dapi/v1/klines"
                f"?symbol={symbol}&interval={interval}&limit={atr_period + 5}"
            )
            r = requests.get(klines_url, timeout=5)
        else:
            klines_url = (
                f"https://fapi.binance.com/fapi/v1/klines"
                f"?symbol={symbol}&interval={interval}&limit={atr_period + 5}"
            )
            r = requests.get(klines_url, timeout=5)
            if r.status_code != 200:
                klines_url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={symbol}&interval={interval}&limit={atr_period + 5}"
                )
                r = requests.get(klines_url, timeout=5)
        klines = r.json()
        if not isinstance(klines, list) or len(klines) < 2:
            return current_price, None, "Could not fetch enough candles for ATR calculation."
    except Exception as e:
        return current_price, None, f"Failed to fetch OHLCV: {e}"

    # Calculate ATR (Wilder's smoothing)
    trs = []
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        prev_close = float(klines[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    if len(trs) < atr_period:
        return current_price, None, "Not enough candles to compute ATR."

    # Simple ATR (mean of last N TRs — good enough for parameter calculation)
    atr = statistics.mean(trs[-atr_period:])
    return current_price, round(atr, 6), None


def fetch_historical_ohlcv(
    symbol: str,
    interval: str,
    days: int = 730,
    progress_callback=None,
    market_type: str = 'usdm',
) -> Tuple[List, Optional[str]]:
    """
    Fetch historical OHLCV candles from Binance (futures then spot fallback).
    Returns (list of raw klines, error_message).
    Each kline: [open_time, open, high, low, close, volume, ...]
    """
    try:
        import requests
    except ImportError:
        return [], "requests not installed"

    symbol = symbol.upper().replace("/", "").replace("-", "")

    # ms per candle
    interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
        "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
        "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
        "1w": 604_800_000,
    }.get(interval, 3_600_000)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 86_400_000)

    all_klines = []
    batch_size = 1500
    current_start = start_ms

    # Select endpoint based on market type
    if market_type == 'coinm':
        base_urls = ["https://dapi.binance.com/dapi/v1/klines"]
    else:
        base_urls = [
            "https://fapi.binance.com/fapi/v1/klines",
            "https://api.binance.com/api/v3/klines",
        ]
    working_url = None
    for url in base_urls:
        try:
            r = requests.get(url, params={
                "symbol": symbol, "interval": interval,
                "startTime": current_start, "limit": 1,
            }, timeout=5)
            if isinstance(r.json(), list):
                working_url = url
                break
        except Exception:
            continue

    if not working_url:
        return [], f"Symbol {symbol} not found on Binance futures or spot."

    batch_num = 0
    total_batches = max(1, (days * 86_400_000) // (batch_size * interval_ms))

    while current_start < end_ms:
        try:
            r = requests.get(working_url, params={
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "limit": batch_size,
            }, timeout=10)
            batch = r.json()
        except Exception as e:
            return all_klines, f"Fetch error at batch {batch_num}: {e}"

        if not isinstance(batch, list) or len(batch) == 0:
            break

        all_klines.extend(batch)
        last_open_time = int(batch[-1][0])
        current_start = last_open_time + interval_ms
        batch_num += 1

        if progress_callback:
            progress_callback(min(1.0, batch_num / total_batches))

        if len(batch) < batch_size:
            break

    return all_klines, None


def klines_to_candles(klines: list):
    """Convert raw Binance klines to Candle objects."""
    from backtest_engine import Candle
    return [
        Candle(
            timestamp=int(k[0]),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
        )
        for k in klines
    ]


def get_available_symbols(limit: int = 50) -> list:
    """
    Return USDT futures symbols from Binance, priority list first (A-Z),
    then remaining symbols A-Z. Only includes symbols with a live price
    on Binance Futures — anything not found is silently dropped.
    """
    PRIORITY = [
        "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT",
        "FARTCOINUSDT", "JTOUSDT", "JUPUSDT", "NEARUSDT", "ONDOUSDT",
        "ORDIUSDT", "SOLUSDT", "SUIUSDT", "XAUUSDT", "XRPUSDT",
    ]
    FALLBACK = sorted(PRIORITY)  # used if API is unreachable

    try:
        import requests
        # Single call: all live Binance Futures tickers
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=5)
        all_tickers = {item["symbol"] for item in r.json() if item["symbol"].endswith("USDT")}

        # Only return symbols from the priority list that are actually live
        return sorted([s for s in PRIORITY if s in all_tickers])

    except Exception:
        return FALLBACK


# ── CoinM (Inverse Perpetual) helpers ────────────────────────────────────────

def get_coinm_symbols() -> list:
    """Return CoinM symbols derived from the live USDM list (always in sync)."""
    return [s.replace('USDT', 'USD_PERP') for s in get_available_symbols()]


def coinm_contract_size(symbol: str) -> int:
    """Contract size in USD: BTC = $100, all others = $10."""
    base = coinm_base_asset(symbol)
    return 100 if base == 'BTC' else 10


def coinm_base_asset(symbol: str) -> str:
    """Extract base asset from CoinM symbol, e.g. BTCUSD_PERP -> BTC."""
    return symbol.replace('USD_PERP', '').replace('USDT', '')
