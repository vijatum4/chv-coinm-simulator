"""CHV Recovery Trade Simulator — Streamlit app."""

import json
import os
import pathlib
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Backtest history directory ────────────────────────────────────────────────
BACKTEST_DIR = pathlib.Path(__file__).parent / "backtests"
BACKTEST_DIR.mkdir(exist_ok=True)


def save_backtest_csv(df: pd.DataFrame, symbol: str, meta: dict) -> pathlib.Path:
    """Save a backtest cycle log CSV with a timestamped filename."""
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"chv_backtest_{symbol}_{ts}.csv"
    fpath = BACKTEST_DIR / fname
    # Prepend a metadata comment row then the data
    with open(fpath, "w") as f:
        f.write("# " + json.dumps(meta) + "\n")
        f.write(df.to_csv(index=False))
    return fpath

def generate_backtest_pdf(result, symbol, tf_display, atr_display, atr_period,
                          efficiency_buffer, reward_ratio, base_lots, leverage,
                          capital, bt_days, fee_rate) -> bytes:
    """Generate a clean PDF summary of a backtest result."""
    from fpdf import FPDF
    from collections import Counter

    def s(text):
        """Sanitize to ASCII-safe characters for Helvetica font."""
        return (str(text)
                .replace("—", "-").replace("–", "-")
                .replace("·", ".").replace("×", "x")
                .replace("↑", "+").replace("↓", "-")
                .replace("•", "-"))

    total_trades = sum(len(c.steps) for c in result.cycles)
    growth_pct = (result.total_net_pnl / capital * 100) if capital else 0
    worst_cycle_idx = next(
        (i for i, c in enumerate(result.cycles) if c.cycle_num == result.worst_intra_loss_cycle), None
    )
    cap_at_worst = (capital + sum(c.net_pnl for c in result.cycles[:worst_cycle_idx])) if worst_cycle_idx is not None else capital
    top6_ws = sorted([c.whipsaws for c in result.cycles], reverse=True)[:6]

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_draw_color(220, 220, 220)

    def divider():
        pdf.ln(3)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)

    def section(title):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, title, ln=True)
        pdf.ln(1)

    # ── Header ───────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "CHV Recovery Trade", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Backtest Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, s(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"), ln=True)
    divider()

    # ── Trade Setup ──────────────────────────────────────────────────────────
    section("TRADE SETUP")
    col_w = 58
    rows = [
        [("Symbol", symbol), ("Trading TF", tf_display), ("ATR TF", atr_display)],
        [("ATR Period", str(atr_period)), ("Buffer", str(efficiency_buffer)), ("Reward Ratio", f"{reward_ratio:.1f}")],
        [("Base Lots", str(base_lots)), ("Leverage", f"{leverage}x"), ("Capital", f"${capital:,.0f}")],
        [("Lookback", f"{bt_days}d"), ("Fee Rate", f"{fee_rate*100:.3f}%"), ("", "")],
    ]
    for row in rows:
        for label, _ in row:
            if label:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(130, 130, 130)
                pdf.cell(col_w, 5, s(label))
        pdf.ln(5)
        for _, value in row:
            if value:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(col_w, 6, s(value))
        pdf.ln(8)
    divider()

    # ── Summary Metrics ──────────────────────────────────────────────────────
    section("SUMMARY")
    metrics = [
        ("Total Cycles",          f"{result.total_cycles:,}",          f"{total_trades:,} trades"),
        ("Win Rate",              f"{result.winning_cycles/result.total_cycles*100:.0f}%" if result.total_cycles else "-",
                                  f"{result.winning_cycles}/{result.total_cycles}"),
        ("Total Net P&L",         f"${result.total_net_pnl:,.2f}",     f"{growth_pct:+.1f}% on ${capital:,.0f}"),
        ("Avg Whipsaws / Cycle",  str(result.avg_whipsaws),            ""),
        ("Top 6 Max Whipsaws",    str(top6_ws[0]) if top6_ws else "0", " / ".join(str(w) for w in top6_ws[1:])),
        ("Worst Intra-Cycle Loss",f"${result.worst_intra_loss:,.2f}",  f"Cycle {result.worst_intra_loss_cycle} / Capital then: ${cap_at_worst:,.2f}"),
        ("Total Fees Paid",       f"${result.total_fees:,.2f}",        f"{fee_rate*100:.3f}% per trade / {total_trades:,} trades"),
    ]
    lw, vw, sw = 52, 38, 90   # label / value / sub column widths
    for i, (label, value, sub) in enumerate(metrics):
        if i % 2 == 0:
            pdf.set_fill_color(248, 248, 248)
        else:
            pdf.set_fill_color(255, 255, 255)
        # label
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(lw, 5, s(label), fill=True)
        # value (bold, larger)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(vw, 5, s(value), fill=True)
        # sub detail
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(sw, 5, s(sub), fill=True, ln=True)
        pdf.ln(1)
    pdf.ln(3)

    # Status banner
    pdf.set_font("Helvetica", "B", 10)
    if result.liquidated:
        pdf.set_fill_color(255, 230, 230)
        pdf.set_text_color(180, 0, 0)
        txt = f"  LIQUIDATED at Cycle {result.liquidation_cycle}, Whipsaw #{result.liquidation_step}"
    else:
        pdf.set_fill_color(220, 245, 220)
        pdf.set_text_color(0, 140, 0)
        txt = f"  Backtest ran the full {bt_days}-day period with no liquidation."
    pdf.cell(0, 8, s(txt), ln=True, fill=True)
    pdf.set_text_color(30, 30, 30)
    divider()

    # ── Whipsaw Breakdown ────────────────────────────────────────────────────
    section("WHIPSAW BREAKDOWN")
    all_ws = [c.whipsaws for c in result.cycles]
    ws_counts = Counter(all_ws)
    max_ws_seen = max(all_ws) if all_ws else 0
    total_c = len(all_ws)
    headers = ["Whipsaws", "Count", "%", "Cumulative %"]
    col_widths = [35, 35, 35, 45]

    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 30)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    cumulative = 0
    pdf.set_font("Helvetica", "", 9)
    for n in range(0, max_ws_seen + 1):
        cnt = ws_counts.get(n, 0)
        cumulative += cnt
        row_vals = [str(n), str(cnt), f"{cnt/total_c*100:.1f}%", f"{cumulative/total_c*100:.1f}%"]
        if n % 2 == 0:
            pdf.set_fill_color(250, 250, 250)
        else:
            pdf.set_fill_color(255, 255, 255)
        for val, w in zip(row_vals, col_widths):
            pdf.cell(w, 6, val, border=1, fill=True)
        pdf.ln()

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "CHV Recovery Trade Simulator  |  System Architect: Chitti Vijakkhana", align="C")

    return bytes(pdf.output())


from chv_engine import (
    calculate_params, simulate, optimize, generate_bot_config,
    CHVParams, SimResult,
)
from backtest_engine import run_backtest
from data_fetcher import fetch_historical_ohlcv, klines_to_candles
from data_fetcher import fetch_price_and_atr, get_available_symbols

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CHV Recovery Trade Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.safe { color: #00c853; font-weight: bold; }
.caution { color: #ffab00; font-weight: bold; }
.trap { color: #d50000; font-weight: bold; }
.metric-box {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("CHV Recovery Trade")
    st.caption("Volatility-Capped Hedging Matrix · Chitti Vijakkhana")
    st.divider()

    mode = st.radio(
        "Mode",
        ["📖 Guide", "Simulator", "Backtest", "Optimizer", "Bot Config", "History", "About"],
        help="Guide: how to use the simulator. Simulator: math model of worst case. Backtest: replay real Binance history. Optimizer: find best params. Bot Config: export JSON. History: browse all saved backtests. About: system concept and formula.",
    )
    st.divider()

    # Asset selection
    st.subheader("Asset")
    use_live = st.toggle("Fetch live from Binance", value=True)

    TF_RULE = {
        "1m":  ("5m",  ""),
        "3m":  ("5m",  ""),
        "5m":  ("30m", ""),
        "15m": ("1h",  ""),
        "30m": ("2h",  ""),
        "1h":  ("4h",  ""),
        "2h":  ("4h",  ""),
        "4h":  ("1d",  ""),
        "6h":  ("1d",  ""),
        "8h":  ("1d",  ""),
        "12h": ("3d",  ""),
        "1d":  ("1w",  ""),
    }

    if use_live:
        symbols = get_available_symbols(30)
        symbol = st.selectbox("Symbol", symbols, index=0)

        trading_tf = st.selectbox(
            "Trading Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"],
            index=4,
            help="The candle TF you'll trade on. ATR TF will be set automatically.",
        )
        recommended_atr_tf, atr_note = TF_RULE.get(trading_tf, ("1d", ""))
        st.caption(f"ATR Timeframe: **{recommended_atr_tf}** {atr_note}")
        atr_tf = recommended_atr_tf
        atr_period = st.slider("ATR Period", 5, 30, 5)

        if st.button("Fetch Price + ATR", type="primary"):
            with st.spinner(f"Fetching {symbol} from Binance..."):
                price, atr, err = fetch_price_and_atr(symbol, atr_tf, atr_period)
            if err:
                st.error(f"Error: {err}")
                st.session_state.pop("live_price", None)
                st.session_state.pop("live_atr", None)
            else:
                st.session_state["live_price"] = price
                st.session_state["live_atr"] = atr
                st.success(f"Price: {price:,.4f}  |  ATR ({atr_tf}): {atr:,.4f}")

        price_val = st.session_state.get("live_price", 85000.0)
        atr_val = st.session_state.get("live_atr", 2500.0)
    else:
        symbol = st.text_input("Symbol", value="BTCUSDT").upper()
        price_val = st.number_input("Current Price", value=85000.0, min_value=0.0001, format="%.4f")
        atr_val = st.number_input("ATR (higher TF)", value=2500.0, min_value=0.0001, format="%.4f")

    st.divider()
    st.subheader("Strategy Parameters")
    efficiency_buffer = st.select_slider(
        "Efficiency Buffer",
        options=[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        value=0.80,
        help="0.8 = standard, 0.9 = high-volume environment",
    )
    reward_ratio = st.slider(
        "Reward Ratio (D = ratio × C)",
        min_value=2.0, max_value=4.0, value=2.5, step=0.1,
        format="%.1f",
        help="2.0 = breakeven baseline (no fees/slippage). 2.5 = standard. Higher = larger TP distance, fewer whipsaws needed to profit.",
    )

    st.divider()
    st.subheader("Position & Capital")
    base_asset = symbol.replace("USDT", "").replace("BUSD", "")

    # Per-symbol Binance lot size constraints
    _LOT_SPECS = {
        "AVAXUSDT":     (1.0,   1.0,   0),
        "BNBUSDT":      (0.01,  0.01,  2),
        "BTCUSDT":      (0.001, 0.001, 3),
        "DOGEUSDT":     (1.0,   1.0,   0),
        "ETHUSDT":      (0.001, 0.001, 3),
        "FARTCOINUSDT": (0.1,   0.1,   1),
        "JTOUSDT":      (1.0,   1.0,   0),
        "JUPUSDT":      (1.0,   1.0,   0),
        "NEARUSDT":     (1.0,   1.0,   0),
        "ONDOUSDT":     (0.1,   0.1,   1),
        "SOLUSDT":      (0.01,  0.01,  2),
        "SUIUSDT":      (0.1,   0.1,   1),
        "XAUUSDT":      (0.001, 0.001, 3),
        "XRPUSDT":      (0.1,   0.1,   1),
    }
    _min_qty, _step, _dec = _LOT_SPECS.get(symbol, (0.001, 0.001, 3))
    _fmt = f"%.{_dec}f"
    _default = max(1.0, _min_qty)

    base_lots = st.number_input(
        f"Base Position Size ({base_asset})",
        value=_default,
        min_value=_min_qty,
        step=_step,
        format=_fmt,
        help=f"Binance min: {_min_qty} {base_asset} · step: {_step}. Scales ×1.5 on each reversal.",
    )
    # ── Binance minimum notional check ───────────────────────────────────────
    _MIN_NOTIONAL = 5.0
    _inv_lots     = base_lots * 0.5
    _inv_notional = _inv_lots * price_val
    if price_val > 0:
        if _inv_notional < _MIN_NOTIONAL:
            import math
            _min_base = math.ceil(_MIN_NOTIONAL * 2 / price_val / _step) * _step
            st.warning(
                f"⚠️ First inversion = {_inv_lots:.{_dec}f} {base_asset} ≈ ${_inv_notional:.2f} "
                f"— below Binance $5 minimum notional. "
                f"Suggest at least **{_min_base:.{_dec}f} {base_asset}** base lots."
            )
        else:
            st.caption(f"✓ First inversion: {_inv_lots:.{_dec}f} {base_asset} ≈ ${_inv_notional:.2f}")
    else:
        st.caption("Fetch live price to validate lot size against Binance $5 minimum.")
    leverage = st.slider("Leverage", 1, 125, 10)
    capital = st.number_input("Starting Capital (USD)", value=1000.0, min_value=1.0, format="%.2f")
    fee_rate_pct = st.number_input(
        "Taker Fee Rate (%)",
        value=0.05,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        help="Enter as percentage. Binance Futures standard = 0.05%. With BNB discount = 0.04%.",
    )
    fee_rate = fee_rate_pct / 100.0

    _slip_options = {
        "0.00% — No slippage (limit orders)": 0.0,
        "0.05% — Liquid pairs (BTC / ETH)": 0.0005,
        "0.10% — Standard alts": 0.001,
        "0.20% — Volatile / small coins": 0.002,
    }
    _slip_label = st.selectbox(
        "Slippage per Fill",
        list(_slip_options.keys()),
        index=1,
        help="Added to every trade as an extra cost on top of fees. Models the difference between your trigger price and actual fill price.",
    )
    slippage_pct = _slip_options[_slip_label]

    st.divider()
    ws_limit_on = st.toggle("Max Whipsaw Limit", value=False,
        help="If ON, cycle is closed at a loss when the whipsaw limit is hit, then waits for the next entry signal.")
    if ws_limit_on:
        ws_limit = st.number_input("Max Whipsaws Before Stop", min_value=1, max_value=50, value=7, step=1,
            help="When a cycle reaches this many whipsaws, the position is closed and the cycle is abandoned.")
    else:
        ws_limit = 0  # 0 = no limit

    st.divider()
    atr_guard_on = st.toggle("ATR Guard", value=True,
        help="If ON, skips entry when footprint (c+d) exceeds ATR × multiplier. Avoids entering in flat/dead markets where the bracket is too large to resolve.")
    if atr_guard_on:
        atr_guard_multiplier = st.slider(
            "Guard Multiplier",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            format="%.1fx",
            help="Entry blocked when footprint > ATR × multiplier. 1.0 = strict (footprint must fit within ATR). 1.5 = lenient (allows up to 1.5× ATR).",
        )
    else:
        atr_guard_multiplier = 1.0  # value unused when guard is off

    max_steps = 50  # run until resolution — the math guarantees it

    st.divider()
    run = st.button("Run Simulation", type="primary", use_container_width=True)


# ── Calculate params always ──────────────────────────────────────────────────

params = calculate_params(symbol, price_val, atr_val, efficiency_buffer, reward_ratio, atr_guard_multiplier)

# ── Lot Size Calculator (sidebar injection — needs params.c) ─────────────────

with st.sidebar:
    st.divider()
    st.subheader("🧮 Lot Size Calculator")
    if not params.is_safe or params.c <= 0:
        st.caption("Set a valid price + ATR first to enable the calculator.")
    else:
        calc_ws = st.number_input(
            "Target Whipsaws to Survive",
            min_value=1, max_value=50, value=10, step=1,
            help="How many whipsaws do you want your capital to be able to fund before running out?",
        )
        # ATR stress-test multiplier
        atr_mult = st.select_slider(
            "ATR Stress Multiplier",
            options=[1.0, 1.5, 2.0, 2.5, 3.0],
            value=1.0,
            format_func=lambda x: f"{x:.1f}×",
            help=(
                "The calculator uses the ATR you fetched right now. "
                "Historical ATR can be much higher — especially on volatile coins at higher price levels. "
                "1.0× = current ATR only. 2.0× = plan for ATR doubling. "
                "Recommended: use 2×–3× for coins with large price swings in your backtest window."
            ),
        )

        # Compute accumulated loss for 1.0 base lot at calc_ws whipsaws
        def _calc_rec_lots(c_val):
            loss = 0.0
            lots = 1.0
            for i in range(1, calc_ws + 1):
                loss += lots * c_val * leverage
                lots = round(0.5, 6) if i == 1 else round(lots * 1.5, 6)
            return round(capital / loss, 6) if loss > 0 else 0.0

        _c_base   = params.c
        _c_stress = params.c * atr_mult
        rec_lots  = _calc_rec_lots(_c_base)
        rec_lots_stress = _calc_rec_lots(_c_stress) if atr_mult > 1.0 else None

        # Current ATR assumption disclosure
        st.caption(
            f"Using ATR = **{atr_val:.6f}** · c = **{_c_base:.6f}** "
            f"(price {price_val:,.4f} · buffer {efficiency_buffer})"
        )

        st.metric(
            "Max Safe Lot Size",
            f"{rec_lots:.4f} {base_asset}",
            help=(
                f"Maximum base lots so your ${capital:,.0f} survives {calc_ws} whipsaws "
                f"at {leverage}× leverage, assuming current ATR {atr_val:.6f}."
            ),
        )
        st.caption(f"Set **Base Position Size ≤ {rec_lots:.4f}** to fund {calc_ws} whipsaws.")

        if rec_lots_stress is not None:
            delta_pct = (rec_lots_stress - rec_lots) / rec_lots * 100
            st.metric(
                f"At {atr_mult:.1f}× ATR (stress)",
                f"{rec_lots_stress:.4f} {base_asset}",
                delta=f"{delta_pct:.0f}% vs base",
                delta_color="inverse",
                help=(
                    f"If ATR rises to {atr_val * atr_mult:.6f} (i.e. {atr_mult:.1f}× current), "
                    f"your max safe lot size drops to {rec_lots_stress:.4f}. "
                    f"Use this number when your backtest window covers periods of higher price / volatility."
                ),
            )
            if rec_lots_stress < rec_lots * 0.6:
                st.warning(
                    f"⚠️ At {atr_mult:.1f}× ATR the safe lot size drops significantly "
                    f"({rec_lots:.1f} → {rec_lots_stress:.1f}). "
                    f"Consider using the stress lot size as your actual base."
                )


# ── Helper: status badge ─────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    if status == "SAFE":
        return '<span class="safe">✅ SAFE</span>'
    elif status == "CAUTION":
        return '<span class="caution">⚠️ CAUTION</span>'
    else:
        return '<span class="trap">🚫 ATR TRAP</span>'


# ══════════════════════════════════════════════════════════════════════════════
# GUIDE MODE
# ══════════════════════════════════════════════════════════════════════════════

# ── GUIDE PAGE ───────────────────────────────────────────────────────────────
if mode == "📖 Guide":
    st.title("📖 Guide")
    st.caption("How to use the CHV Recovery Trade Simulator")
    st.divider()

    st.header("Getting Started")
    st.markdown("""
1. **Pick your asset** — Use the sidebar to select a symbol and click **Fetch Price + ATR** to load live data from Binance.
2. **Set strategy parameters** — Choose your Efficiency Buffer and Reward Ratio.
3. **Set your position size and capital** — Enter how many coins per trade, your leverage, and your available capital.
4. **Run the Simulator** — See the mathematical worst-case ledger: how many whipsaws your capital survives and what each step costs.
5. **Run a Backtest** — Replay real historical data to see how many cycles occurred, how they resolved, and the full P&L over time.
6. **Use the Optimizer** — Sweep multiple parameter combinations to find the best setup for your capital and risk tolerance.
""")

    st.divider()
    st.header("Sidebar Input Reference")

    st.subheader("🔷 Asset")
    st.markdown("""
<table style="width:100%;border-collapse:collapse">
<thead><tr style="border-bottom:1px solid #333">
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:220px">Input</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal">What it does</th>
</tr></thead>
<tbody>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Fetch live from Binance</span></td><td style="padding:8px 12px">Pulls the current price and ATR directly from Binance. Turn OFF to enter values manually for offline testing.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Symbol</span></td><td style="padding:8px 12px">The trading pair (e.g. BTCUSDT, FARTCOINUSDT). Must be a valid Binance Futures perpetual.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Trading Timeframe</span></td><td style="padding:8px 12px">The candle interval you will trade on. Determines entry/exit timing in backtests. Shorter TF = more cycles, more fees.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">ATR Timeframe</span></td><td style="padding:8px 12px">Auto-set to a higher TF than your trading TF. Uses macro volatility to size the bracket — prevents sizing off short-term noise.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">ATR Period</span></td><td style="padding:8px 12px">Number of candles used to compute ATR. Default 14 is standard. Higher = smoother but slower to adapt to new volatility regimes.</td></tr>
</tbody></table>
""", unsafe_allow_html=True)

    st.subheader("🔷 Strategy Parameters")
    st.markdown("""
<table style="width:100%;border-collapse:collapse">
<thead><tr style="border-bottom:1px solid #333">
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:180px">Input</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:280px">What it does</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal">Effect on the system</th>
</tr></thead>
<tbody>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Efficiency Buffer</span></td><td style="padding:8px 12px;vertical-align:top">Scales the cut-loss zone <code>c</code> as a fraction of ATR. <code>0.80</code> means c = 80% of the ATR value.</td><td style="padding:8px 12px;vertical-align:top"><b>Higher</b> → wider bracket → fewer whipsaws per cycle, but each whipsaw costs more. <b>Lower</b> → tighter bracket → more frequent whipsaws, smaller loss per whipsaw.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Reward Ratio</span></td><td style="padding:8px 12px;vertical-align:top">Sets the TP distance <code>d = ratio × c</code>. How far price needs to move in your favour to close the cycle.</td><td style="padding:8px 12px;vertical-align:top"><b>2.0</b> = mathematical breakeven (no fees). <b>2.5</b> = standard — each TP earns ~20% more than a single whipsaw costs. Higher ratio = larger profit per TP but longer to resolve.</td></tr>
</tbody></table>
""", unsafe_allow_html=True)
    st.info("💡 Buffer and Ratio work together. A tighter buffer with a higher ratio can produce the same bracket shape as a wider buffer with a lower ratio — but with different whipsaw frequency and per-whipsaw cost. The Optimizer sweeps these combinations for you.")

    st.subheader("🔷 Position & Capital")
    st.markdown("""
<table style="width:100%;border-collapse:collapse">
<thead><tr style="border-bottom:1px solid #333">
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:180px">Input</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:280px">What it does</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal">Relationship</th>
</tr></thead>
<tbody>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Base Position Size</span></td><td style="padding:8px 12px;vertical-align:top">Number of coins/contracts in your opening trade.</td><td style="padding:8px 12px;vertical-align:top">All P&L and margin requirements scale linearly with this. Doubling lots doubles both profits and losses at every step.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Leverage</span></td><td style="padding:8px 12px;vertical-align:top">Multiplies market exposure relative to the margin used per position.</td><td style="padding:8px 12px;vertical-align:top">Higher leverage = less margin locked per trade = more whipsaws fundable from the same capital. But also amplifies P&L per step. Use the Simulator to validate.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Starting Capital</span></td><td style="padding:8px 12px;vertical-align:top">Your total available account balance in USD.</td><td style="padding:8px 12px;vertical-align:top">The hard limit on how deep the system can go. The Simulator shows exactly how many whipsaws this capital can fund before margin runs out.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Taker Fee Rate</span></td><td style="padding:8px 12px;vertical-align:top">Exchange fee charged per fill (entry + exit), as a percentage.</td><td style="padding:8px 12px;vertical-align:top">Paid on every position open and close. A cycle with 10 whipsaws = 22 fills = 22× fee. At 0.05%, this compounds noticeably on long cycles.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">Slippage per Fill</span></td><td style="padding:8px 12px;vertical-align:top">Extra cost per fill modelling the gap between your trigger price and actual execution price.</td><td style="padding:8px 12px;vertical-align:top">Negligible for limit orders on liquid pairs (use 0%). Meaningful for market orders or low-liquidity coins — use 0.10–0.20%.</td></tr>
</tbody></table>
""", unsafe_allow_html=True)

    st.subheader("🔷 Max Whipsaw Limit")
    st.markdown("""
<table style="width:100%;border-collapse:collapse">
<thead><tr style="border-bottom:1px solid #333">
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal;width:180px">Setting</th>
  <th style="text-align:left;padding:8px 12px;color:#888;font-weight:normal">Behaviour</th>
</tr></thead>
<tbody>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">OFF</span></td><td style="padding:8px 12px">No limit. Cycle runs until either the TP is hit (full recovery + profit) or capital is exhausted (liquidation). The math guarantees recovery if capital holds.</td></tr>
<tr><td style="padding:8px 12px;vertical-align:top"><span style="color:#4cc9f0;font-weight:bold">ON + limit N</span></td><td style="padding:8px 12px">When a cycle reaches N whipsaws, the active position is closed at a deliberate loss. The system then waits for the next entry signal to start a fresh cycle.</td></tr>
</tbody></table>
""", unsafe_allow_html=True)
    st.markdown("""

**When to use it:**
Look at the **Whipsaw Breakdown** table from a backtest. If 95% of cycles resolve within 10 whipsaws,
but your capital can only safely fund 12, setting a limit of 12 means you only take a managed loss on the
rarest, deepest cycles — and avoid a full margin blowout.

The trade-off: you will occasionally stop out a cycle that would have recovered on whipsaw 13 or 14.
""")

    st.divider()
    st.header("Understanding the Results")
    st.markdown("""
**Simulator → Whipsaw Ledger**
Shows every possible step from your entry to TP, including every inversion. The danger step (red) is where
your capital runs out. If there is no red row, your capital covers the full depth.

**Backtest → Summary Metrics**
- **Win Rate** — % of cycles that hit TP (vs. stopped or liquidated)
- **Avg Whipsaws/Cycle** — typical depth of a cycle on this asset/TF
- **Top 6 Max Whipsaws** — the worst cycles in the data, to size your capital buffer
- **Worst Intra-Cycle Loss** — the deepest unrealized loss before a TP was hit — this is your real drawdown exposure

**Backtest → Whipsaw Breakdown Table**
The full distribution of whipsaw counts across all cycles.
Cross-reference with your capital limit (from the Simulator) to understand what % of historical
cycles would have survived your setup without hitting the limit.

**Rule of thumb:** Fund your capital to survive the 90th percentile whipsaw count + 3 extra as buffer.
""")

    st.stop()


# ── ABOUT PAGE ───────────────────────────────────────────────────────────────
if mode == "About":
    st.title("CHV Recovery Trade System")
    st.caption("Volatility-Capped Hedging Matrix · Chitti Vijakkhana")
    st.divider()

    st.header("What is the CHV System?")
    st.markdown("""
The **CHV Recovery Trade** is a structured hedging strategy built around a simple mathematical truth:
if you know exactly how far price needs to move to hit your target, and you scale positions precisely,
**every cycle is guaranteed to recover — regardless of how many times price reverses.**

Most trading systems treat a stop-loss as a finalized loss. CHV treats it as an **inversion signal** —
the moment price exits the bracket in the wrong direction, the position is flipped rather than closed.
The lot size at each inversion is calculated so that when the target is finally hit,
it pays back all previous losses and returns the intended profit on top.

> **Core idea:** Let the market whipsaw. The math works in your favour every time price eventually trends.
""")

    st.divider()

    st.header("The Price Bracket")
    st.markdown("""
Every CHV cycle is anchored to **four price levels** derived from the current price and ATR:

| Level | Name | Role |
|---|---|---|
| **TL** | Take Long | Long take-profit — price must rise here to close a winning long |
| **LP** | Long Price | Long entry — where the initial long position is opened |
| **SP** | Short Price | Short entry / long stop-loss — crossing here triggers inversion |
| **TS** | Take Short | Short take-profit — price must fall here to close a winning short |

```
TL  ──────────────────────────  ← Long TP        ( LP + d )
LP  ──────────────────────────  ← Long entry / Short SL
SP  ──────────────────────────  ← Short entry / Long SL   ( LP − c )
TS  ──────────────────────────  ← Short TP       ( SP − d )
```

The bracket is **symmetrical** — `c` is the risk distance, `d` is the reward distance.
""")

    st.divider()

    st.header("The Formula")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bracket Sizing")
        st.markdown("""
```
c  =  ATR  ×  efficiency_buffer
d  =  reward_ratio  ×  c

LP  =  entry_price
SP  =  LP − c
TL  =  LP + d
TS  =  SP − d
```
- **c** — the cut-loss zone: width of one adverse move
- **d** — the recovery target: distance to take-profit
- Ratio **2.0** = mathematical breakeven (zero fees)
- Ratio **2.5+** = profit margin built in above breakeven
""")

    with col2:
        st.subheader("Lot Scaling on Inversion")
        st.markdown("""
```
1st inversion:   lots = base_lots × 0.5
2nd inversion:   lots = prev_lots × 1.5
3rd inversion:   lots = prev_lots × 1.5
       ⋮
```
This is not a tuned parameter — it is the **exact multiplier** derived
from the requirement that TP profit covers all accumulated losses plus
the original target profit `d`:

```
new_lots × d  =  accumulated_loss + base_lots × d
```
Solving this gives **x = 1.5** and first inversion factor **x − 1 = 0.5**.
""")

    st.info("💡 **Breakeven proof:** With ratio = 2.0 and zero fees, any number of whipsaws nets exactly $0. Every unit of ratio above 2.0 converts into pure profit per cycle.")

    st.divider()

    st.header("What Makes CHV Different")
    u1, u2, u3 = st.columns(3)
    with u1:
        st.markdown("""
**🔄 No Realized Losses**

Traditional systems close losing positions. CHV inverts them.
Every SL is an open hedge — the loss is unrealized until the TP
recovers it, with profit.
""")
    with u2:
        st.markdown("""
**📐 Mathematically Guaranteed Recovery**

The lot formula is derived, not tuned. As long as capital survives
to the next TP, the cycle nets positive — not by luck, but by arithmetic.
""")
    with u3:
        st.markdown("""
**📊 ATR-Adaptive Bracket**

Bracket width scales with real volatility. Calm market = tight bracket.
Volatile market = wider bracket. Structure stays proportional to
actual price behaviour.
""")

    u4, u5, u6 = st.columns(3)
    with u4:
        st.markdown("""
**↕️ Directional Neutrality**

Works identically long and short. After a long TP, the next cycle
opens long (momentum continuation). After a short TP, opens short.
Direction follows the market.
""")
    with u5:
        st.markdown("""
**⚖️ Single Risk Variable**

The only thing that ends a cycle badly is running out of capital
to fund the next inversion. Direction, whipsaw count, and duration
are all handled by the math.
""")
    with u6:
        st.markdown("""
**🛡️ Bounded Risk Option**

The optional whipsaw limit lets you cap exposure. Instead of risking
a margin blowout at deep whipsaw counts, you take a controlled loss
and re-enter on the next signal.
""")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR MODE
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Simulator" and not run:
    st.title(f"CHV Simulator — {symbol}")
    st.info("Set your parameters in the sidebar, then click **Run Simulation**.")
    st.stop()

if mode == "Simulator":

    # ── Parameter summary ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Volatility Parameters")
        st.metric("ATR (Macro)", f"{params.atr_macro:,.4f}")
        st.metric("Efficiency Buffer", f"{params.efficiency_buffer}")
        st.metric("Cut Loss Zone (c)", f"{params.c:,.4f}")
        st.metric("Recovery Target (d)", f"{params.d:,.4f}")
        st.metric("Footprint (c+d)", f"{params.footprint:,.4f}  ({params.footprint_pct*100:.1f}% of ATR)")

    with col2:
        st.subheader("Operational Levels")
        st.metric("LP  Long Entry", f"{params.lp:,.4f}")
        st.metric("SP  Short Entry", f"{params.sp:,.4f}")
        st.metric("TL  Long TP", f"{params.tl:,.4f}")
        st.metric("TS  Short TP", f"{params.ts:,.4f}")

    with col3:
        st.subheader("Structure Check")
        st.markdown(status_badge(params.safety_status), unsafe_allow_html=True)
        st.caption(params.safety_message)
        st.metric("Lot Multiplier", "1.5× (fixed)")
        st.metric("Leverage", f"{leverage}×")
        st.metric("Fee Rate", f"{fee_rate*100:.2f}%")

    st.divider()

    if not params.is_safe:
        st.error(f"ATR TRAP DETECTED — DO NOT EXECUTE\n\n{params.safety_message}")
        st.stop()

    # ── Run simulation ───────────────────────────────────────────────────────
    result: SimResult = simulate(params, base_lots, leverage, capital, max_steps, fee_rate)

    # ── Capital sustainability banner ────────────────────────────────────────
    cap_cols = st.columns(3)
    cap_cols[0].metric(
        "Max Whipsaws Your Capital Can Sustain",
        f"{result.steps_covered} steps",
        help="The system will always resolve — this tells you how many reversals your capital can fund before running dry.",
    )
    cap_cols[1].metric("Peak Margin Required", f"${result.peak_margin:,.2f}", f"at step {result.peak_margin_step}")
    cap_cols[2].metric("Resolved at Step", str(result.resolution_step) if result.resolved else "—")

    if not result.capital_survives:
        st.error(
            f"Capital runs out at step {result.danger_step} "
            f"(${result.danger_margin:,.2f} needed, ${capital:,.2f} available). "
            "Increase capital, reduce base lots, or lower leverage."
        )
    elif result.steps_covered < 5:
        st.warning("Capital sustains fewer than 5 whipsaws — consider adding more capital or reducing position size.")

    # ── Ledger table ─────────────────────────────────────────────────────────
    st.subheader("Whipsaw Ledger")

    rows = []
    for s in result.steps:
        rows.append({
            "Step": s.step,
            "Direction": s.direction,
            "Trigger Price": f"{s.trigger_price:,.4f}",
            "Active Lots": f"{s.active_lots:.4f}" if not s.is_exit else "—",
            "Leg PnL": f"${s.closed_pnl:,.2f}",
            "Running Balance": f"${s.running_balance:,.2f}",
            "Margin Required": f"${s.margin_required:,.2f}" if not s.is_exit else "—",
        })

    df = pd.DataFrame(rows)

    def highlight_row(row):
        styles = [""] * len(row)
        if row["Direction"] == "EXIT":
            styles = ["background-color: #1b4332; color: #52b788"] * len(row)
        elif "LONG" in row["Direction"]:
            styles = ["background-color: #0d1b2a"] * len(row)
        else:
            styles = ["background-color: #1a0a0a"] * len(row)
        # Danger zone
        if result.danger_step and row["Step"] == result.danger_step:
            styles = ["background-color: #4a0000; color: #ff6b6b"] * len(row)
        return styles

    st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.subheader("Equity Curve + Lot Scaling")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Cumulative P&L (USD)", "Active Lots per Step"),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    step_nums = [s.step for s in result.steps]
    balances = [s.running_balance for s in result.steps]
    lots_list = [s.active_lots for s in result.steps]
    colors = [
        "#52b788" if s.is_exit else ("#ef233c" if b < 0 else "#4cc9f0")
        for s, b in zip(result.steps, balances)
    ]

    fig.add_trace(
        go.Bar(x=step_nums, y=balances, marker_color=colors, name="P&L"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=step_nums, y=lots_list,
            mode="lines+markers",
            line=dict(color="#f4a261", width=2),
            marker=dict(size=6),
            name="Lots",
        ),
        row=2, col=1,
    )

    if result.danger_step:
        fig.add_vline(
            x=result.danger_step, line_dash="dot",
            line_color="#d00000", annotation_text="Capital Limit",
            row=1, col=1,
        )

    fig.update_layout(
        height=500,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#1e1e2e")

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Net P&L (after all fees)", f"${result.net_pnl:,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER MODE
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "Optimizer":
    st.title(f"CHV Optimizer — {symbol}")
    st.info(
        "The optimizer sweeps all combinations of efficiency buffer, leverage, and base lot size "
        "to find the setup that produces the best net yield while keeping your capital safe."
    )

    def price_aware_lot_defaults(price: float, capital: float) -> list:
        """Return 4 lot sizes spanning ~$1 to ~3% of capital in notional value."""
        import math
        if price <= 0:
            return [0.001, 0.01, 0.1, 1.0]
        targets = [1.0, 5.0, 25.0, max(50.0, capital * 0.03)]
        lots = []
        for usd in targets:
            raw = usd / price
            if raw < 0.00001:
                continue
            mag = 10 ** math.floor(math.log10(raw))
            rounded = round(round(raw / mag) * mag, 6)
            if rounded > 0 and rounded not in lots:
                lots.append(rounded)
        return lots if lots else [0.001, 0.01, 0.1, 1.0]

    suggested_lots = price_aware_lot_defaults(price_val, capital)
    suggested_str = ", ".join(str(x) for x in suggested_lots)

    # ── Helper: minimum capital to survive N whipsaws ─────────────────────────
    def min_capital_for_whipsaws(base_lots, entry_price, c, leverage, target_n):
        lots = base_lots
        accumulated_loss = 0.0
        for _ in range(target_n):
            accumulated_loss += lots * c * leverage
            lots = round(lots * 1.5, 6)
        margin_next = (lots * entry_price) / leverage
        return round(max(accumulated_loss, margin_next), 2)

    # ── Preset TF pairs for sweep ─────────────────────────────────────────────
    TF_SWEEP_PAIRS = {
        "1m / 5m ATR":   ("1m",  "5m"),
        "5m / 30m ATR":  ("5m",  "30m"),
        "15m / 1h ATR":  ("15m", "1h"),
        "30m / 2h ATR":  ("30m", "2h"),
        "1h / 4h ATR":   ("1h",  "4h"),
        "4h / 1d ATR":   ("4h",  "1d"),
        "12h / 3d ATR":  ("12h", "3d"),
        "1d / 1w ATR":   ("1d",  "1w"),
    }

    with st.expander("Sweep Ranges", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            buffers = st.multiselect(
                "Efficiency Buffers",
                [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                default=[0.75, 0.80, 0.85, 0.90],
            )
        with col2:
            leverages = st.multiselect(
                "Leverage Values",
                [1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100],
                default=[5, 10, 20, 25],
            )
        with col3:
            lot_input = st.text_input(
                "Base Lot Sizes (comma-separated)",
                value=suggested_str,
                help=f"Auto-suggested for {symbol} at ${price_val:,.4f}. Each value = number of coins in your starting position.",
            )
            try:
                lot_sizes = [float(x.strip()) for x in lot_input.split(",") if x.strip()]
            except ValueError:
                lot_sizes = suggested_lots
                st.warning("Invalid lot sizes, using suggested defaults.")
            if price_val > 0 and lot_sizes:
                notionals = [f"{x} coins ≈ ${x * price_val:,.2f}" for x in lot_sizes]
                st.caption("Notional: " + " · ".join(notionals))

    with st.expander("TF Sweep — test multiple timeframe combinations", expanded=False):
        sweep_tfs = st.multiselect(
            "Timeframe combinations to test",
            list(TF_SWEEP_PAIRS.keys()),
            default=["1h / 4h ATR", "4h / 1d ATR"],
            help="The optimizer will fetch the ATR for each TF pair and run the full sweep. Results will show which TF combo scores highest.",
        )
        enable_tf_sweep = len(sweep_tfs) > 0
        st.caption("If no TF selected, uses the ATR already fetched in the sidebar.")

    with st.expander("Capital & Whipsaw Target", expanded=True):
        target_whipsaws = st.number_input(
            "Target whipsaws to survive",
            min_value=1, max_value=30, value=5,
            help="The optimizer will calculate the minimum capital required for each setup to survive this many whipsaw rounds without liquidation.",
        )

    if st.button("Run Optimizer", type="primary"):
        # Build list of (trading_tf, atr_tf, atr_value) to sweep
        tf_atr_combos = []
        if enable_tf_sweep:
            with st.spinner("Fetching ATR for selected timeframes..."):
                for label in sweep_tfs:
                    t_tf, a_tf = TF_SWEEP_PAIRS[label]
                    _, atr_fetched, err = fetch_price_and_atr(symbol, a_tf, atr_period)
                    if err or not atr_fetched:
                        st.warning(f"Could not fetch ATR for {label}: {err}")
                    else:
                        tf_atr_combos.append((t_tf, a_tf, atr_fetched))
        else:
            tf_atr_combos = [("—", "—", atr_val)]

        all_rows = []
        for t_tf, a_tf, atr_for_sweep in tf_atr_combos:
            combos = len(buffers) * len(leverages) * len(lot_sizes)
            with st.spinner(f"Testing {combos} combos for {t_tf}/{a_tf} ATR={atr_for_sweep:.4f}..."):
                sweep_results = optimize(
                    symbol=symbol,
                    price=price_val,
                    atr=atr_for_sweep,
                    capital=capital,
                    fee_rate=fee_rate,
                    buffers=buffers,
                    leverages=leverages,
                    lot_sizes=lot_sizes,
                    max_steps=max_steps,
                )
            entry_price = price_val
            for r in sweep_results:
                from chv_engine import calculate_params
                p = calculate_params(symbol, price_val, atr_for_sweep, r.buffer, reward_ratio)
                min_cap = min_capital_for_whipsaws(r.base_lots, entry_price, p.c, r.leverage, target_whipsaws)
                all_rows.append({
                    "Trading TF": t_tf,
                    "ATR TF": a_tf,
                    "Buffer": r.buffer,
                    "Leverage": f"{r.leverage}×",
                    "Base Lots": r.base_lots,
                    "Net PnL": f"${r.net_pnl:,.2f}",
                    "Steps Covered": r.steps_covered,
                    "Capital Safe": "✅" if r.capital_survives else "❌",
                    "Footprint %": f"{r.footprint_pct*100:.1f}%",
                    "Status": r.safety_status,
                    f"Min Capital ({target_whipsaws} WS)": f"${min_cap:,.2f}",
                    "Score": r.score,
                    "_score": r.score,
                    "_buf": r.buffer,
                    "_lev": r.leverage,
                    "_lots": r.base_lots,
                    "_pnl": r.net_pnl,
                    "_ttf": t_tf,
                    "_atf": a_tf,
                })

        if not all_rows:
            st.error("No safe setups found across all selected timeframes.")
        else:
            all_rows.sort(key=lambda x: x["_score"], reverse=True)
            top = all_rows[:25]
            display_cols = [k for k in top[0].keys() if not k.startswith("_")]
            df = pd.DataFrame(top)[display_cols]
            st.success(f"Found {len(all_rows)} viable setups. Showing top 25.")
            st.dataframe(df, use_container_width=True, hide_index=True)

            best = top[0]
            st.subheader("Best Setup")
            bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
            bc1.metric("Trading TF", best["Trading TF"])
            bc2.metric("ATR TF", best["ATR TF"])
            bc3.metric("Buffer", best["Buffer"])
            bc4.metric("Leverage", best["Leverage"])
            bc5.metric("Base Lots", best["Base Lots"])
            bc6.metric("Net PnL", best["Net PnL"])
            st.metric(f"Min Capital to survive {target_whipsaws} whipsaws", best[f"Min Capital ({target_whipsaws} WS)"])

            st.session_state["best_buffer"] = best["_buf"]
            st.session_state["best_leverage"] = best["_lev"]
            st.session_state["best_lots"] = best["_lots"]
            st.info("Switch to Simulator or Backtest mode and apply these values to validate the setup.")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST MODE
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "Backtest":
    # ── Trade Setup Summary ──────────────────────────────────────────────────
    tf_display = trading_tf if use_live else "manual"
    atr_display = atr_tf if use_live else "manual"

    sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9 = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 1])
    sc1.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Symbol</p>
<p style="font-size:1.4rem;font-weight:600;color:#4cc9f0;margin:0">{symbol}</p>
</div>""", unsafe_allow_html=True)
    sc2.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Trading TF</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{tf_display}</p>
</div>""", unsafe_allow_html=True)
    sc3.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">ATR TF</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{atr_display}</p>
</div>""", unsafe_allow_html=True)
    sc4.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">ATR Period</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{atr_period}</p>
</div>""", unsafe_allow_html=True)
    sc5.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Buffer</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{efficiency_buffer}</p>
</div>""", unsafe_allow_html=True)
    sc6.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Reward Ratio</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{reward_ratio:.1f}</p>
</div>""", unsafe_allow_html=True)
    sc7.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Base Lots</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{base_lots}</p>
</div>""", unsafe_allow_html=True)
    sc8.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Leverage</p>
<p style="font-size:1.4rem;font-weight:600;color:#fafafa;margin:0">{leverage}×</p>
</div>""", unsafe_allow_html=True)
    sc9.markdown(f"""<div>
<p style="font-size:12px;color:#888;margin:0 0 3px 0">Capital</p>
<p style="font-size:1.4rem;font-weight:600;color:#00c853;margin:0">${capital:,.0f}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:30px'></div>", unsafe_allow_html=True)

    # Lot size warning (only show if clearly over limit)
    if params.c > 0 and leverage > 0:
        loss_per_lot_5steps = sum((1.5 ** n) * params.c * leverage for n in range(1, 6))
        suggested_lots = capital / loss_per_lot_5steps
        if base_lots > suggested_lots * 2:
            st.warning(f"⚠️ {base_lots} {base_asset} lots may be too high — suggested max ~{suggested_lots:.1f} for 5-whipsaw survival.")

    # ── Run row: [Run Backtest] [Lookback] [Enforce min] [Progress bar] ────────
    bcol1, bcol2, bcol3, bcol4 = st.columns([1, 1, 1, 2])
    with bcol1:
        run_bt = st.button("Run Backtest", type="primary")
    with bcol2:
        bt_days = st.selectbox("Lookback", [90, 180, 365, 730], index=2,
                                format_func=lambda x: f"{x}d", label_visibility="collapsed")
    with bcol3:
        enforce_min_notional = st.checkbox(
            "Binance $5 min",
            value=True,
            help="When ON, blocks the backtest if first inversion (base × 0.5 × price) < $5 — simulates real Binance order rejection.",
        )
    prog_slot = bcol4.empty()

    # Enforce minimum notional before running
    if run_bt and enforce_min_notional and price_val > 0:
        import math
        _inv_notional_bt = base_lots * 0.5 * price_val
        if _inv_notional_bt < 5.0:
            _min_base_bt = math.ceil(10.0 / price_val / _step) * _step
            st.error(
                f"❌ Backtest blocked: first inversion = {base_lots * 0.5:.{_dec}f} {base_asset} "
                f"≈ ${_inv_notional_bt:.2f} — below Binance $5 minimum notional. "
                f"Increase base lots to at least **{_min_base_bt:.{_dec}f} {base_asset}**, "
                f"or uncheck **Binance $5 min** to run without enforcement."
            )
            run_bt = False

    if run_bt:
        tf_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h",
            "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w",
        }
        trade_interval = tf_map.get(trading_tf if use_live else "1h", "1h")
        atr_interval = tf_map.get(atr_tf if use_live else "4h", "4h")

        prog_slot.progress(0, text="Fetching trading TF candles...")
        trading_klines, err1 = fetch_historical_ohlcv(
            symbol, trade_interval, days=bt_days,
            progress_callback=lambda p: prog_slot.progress(p * 0.5, text=f"Fetching {trade_interval} candles...")
        )
        if err1:
            st.error(f"Failed to fetch trading candles: {err1}")
            st.stop()

        prog_slot.progress(0.5, text="Fetching ATR TF candles...")
        atr_klines, err2 = fetch_historical_ohlcv(
            symbol, atr_interval, days=bt_days,
            progress_callback=lambda p: prog_slot.progress(0.5 + p * 0.3, text=f"Fetching {atr_interval} candles...")
        )
        if err2:
            st.error(f"Failed to fetch ATR candles: {err2}")
            st.stop()

        prog_slot.progress(0.8, text="Running CHV backtest...")
        trading_candles = klines_to_candles(trading_klines)
        atr_candles = klines_to_candles(atr_klines)

        result = run_backtest(
            symbol=symbol,
            trading_candles=trading_candles,
            atr_candles=atr_candles,
            atr_period=atr_period,
            base_lots=base_lots,
            leverage=leverage,
            capital=capital,
            fee_rate=fee_rate,
            buffer=efficiency_buffer,
            slippage_pct=slippage_pct,
            reward_ratio=reward_ratio,
            max_whipsaws=ws_limit,
            atr_guard=atr_guard_on,
            atr_guard_multiplier=atr_guard_multiplier,
        )
        result.trading_tf = trade_interval
        result.atr_tf = atr_interval
        prog_slot.progress(1.0, text="Done.")
        st.session_state["bt_result"] = result

    result = st.session_state.get("bt_result")
    if result and result.symbol == symbol:
        import datetime

        st.divider()
        st.subheader("Summary")

        # Pre-compute derived metrics
        total_trades = sum(len(c.steps) for c in result.cycles)
        growth_pct = (result.total_net_pnl / capital * 100) if capital else 0
        worst_cycle_idx = next(
            (i for i, c in enumerate(result.cycles) if c.cycle_num == result.worst_intra_loss_cycle), None
        )
        cap_at_worst = (capital + sum(c.net_pnl for c in result.cycles[:worst_cycle_idx])) if worst_cycle_idx is not None else capital

        # Row 1
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Total Cycles", f"{result.total_cycles:,}",
                  delta=f"{total_trades:,} trades",
                  help="Each cycle = one full sequence from entry to TP exit. Total trades counts every individual order placed.")
        m2.metric("Win Rate", f"{result.winning_cycles/result.total_cycles*100:.0f}%" if result.total_cycles else "—",
                  delta=f"{result.winning_cycles}/{result.total_cycles}")
        pnl_color = "#00c853" if result.total_net_pnl >= 0 else "#ef233c"
        growth_color = "#00c853" if growth_pct >= 0 else "#ef233c"
        growth_arrow = "↑" if growth_pct >= 0 else "↓"
        m3.markdown(f"""
<div>
<p style="font-size:14px;color:#888;margin:0 0 4px 0">Total Net P&L</p>
<p style="font-size:2rem;font-weight:600;color:{pnl_color};margin:0">${result.total_net_pnl:,.2f}</p>
<p style="font-size:12px;color:{growth_color};margin:4px 0 0 0">{growth_arrow} {growth_pct:+.1f}% on ${capital:,.0f}</p>
</div>""", unsafe_allow_html=True)
        m4.metric("Avg Whipsaws/Cycle", result.avg_whipsaws)
        top6_ws = sorted([c.whipsaws for c in result.cycles], reverse=True)[:6]
        rest_str = " · ".join(str(w) for w in top6_ws[1:])
        m5.markdown(f"""
<div>
<p style="font-size:14px;color:#888;margin:0 0 4px 0">Top 6 Max Whipsaws</p>
<p style="font-size:2rem;font-weight:600;color:#fafafa;margin:0">{top6_ws[0] if top6_ws else 0}</p>
<p style="font-size:12px;color:#888;margin:4px 0 0 0">{rest_str}</p>
</div>""", unsafe_allow_html=True)
        m6.markdown(f"""
<div>
<p style="font-size:14px;color:#888;margin:0 0 4px 0">Worst Intra-Cycle Loss</p>
<p style="font-size:2rem;font-weight:600;color:#ef233c;margin:0">${result.worst_intra_loss:,.2f}</p>
<p style="font-size:12px;color:#888;margin:4px 0 0 0">cycle {result.worst_intra_loss_cycle} · capital then: <span style="color:#00c853">${cap_at_worst:,.2f}</span></p>
</div>""", unsafe_allow_html=True)
        m7.markdown(f"""
<div>
<p style="font-size:14px;color:#888;margin:0 0 4px 0">Total Fees Paid</p>
<p style="font-size:2rem;font-weight:600;color:#ef233c;margin:0">${result.total_fees:,.2f}</p>
<p style="font-size:12px;color:#888;margin:4px 0 0 0">{fee_rate*100:.3f}% per trade · {total_trades:,} trades</p>
</div>""", unsafe_allow_html=True)

        # Liquidation / completion banner
        if result.liquidated:
            completed = result.total_cycles - 1
            cap_before_liq = capital + sum(c.net_pnl for c in result.cycles[:-1])
            loss_mag = abs(result.liquidation_loss)
            is_margin_liq = loss_mag < cap_before_liq
            if is_margin_liq:
                st.error(f"""
**Account LIQUIDATED at Cycle {result.liquidation_cycle} — Margin Exhausted at Position #{result.liquidation_step}**

- Cycles completed before blowup: **{completed}**
- Capital available: **${cap_before_liq:,.2f}**
- Realized losses before failure: **${result.liquidation_loss:,.2f}**
- Cause: after **{result.liquidation_step - 1} whipsaws**, the next position required more margin than the account held

**What to fix:**
- Reduce base lots (currently {base_lots} {base_asset}) — position grows ×1.5 per whipsaw
- Or increase capital to cover larger position sizes
- Or increase leverage to reduce margin required per position
""")
            else:
                shortfall = loss_mag - cap_before_liq
                st.error(f"""
**Account LIQUIDATED at Cycle {result.liquidation_cycle}, Whipsaw #{result.liquidation_step}**

- Cycles completed before blowup: **{completed}**
- Capital at liquidation cycle start: **${cap_before_liq:,.2f}**
- Loss at liquidation: **${result.liquidation_loss:,.2f}**
- Shortfall: **${shortfall:,.2f}** (accumulated loss exceeded available capital)

**What to fix:**
- Reduce base lots (currently {base_lots} {base_asset}) to survive more whipsaws
- Or increase capital to cover deeper cycles
- Or lower leverage to reduce loss per whipsaw
""")
        elif result.data_exhausted:
            st.success(f"Backtest ran the full {bt_days}-day period with no liquidation — capital held throughout.")

        # Whipsaw Breakdown Table
        st.divider()
        st.subheader("Whipsaw Breakdown")
        all_ws = [c.whipsaws for c in result.cycles]
        total_c = len(all_ws)

        if total_c:
            from collections import Counter
            ws_counts = Counter(all_ws)
            max_ws_seen = max(all_ws)
            ws_rows = []
            cumulative = 0
            for n in range(0, max_ws_seen + 1):
                cnt = ws_counts.get(n, 0)
                cumulative += cnt
                pct = cnt / total_c * 100
                cum_pct = cumulative / total_c * 100
                ws_rows.append({
                    "Whipsaws": n,
                    "Count": cnt,
                    "%": f"{pct:.1f}%",
                    "Cumulative %": f"{cum_pct:.1f}%",
                })
            df_ws = pd.DataFrame(ws_rows)

            def style_ws_rows(row):
                n = row["Whipsaws"]
                cnt = ws_counts.get(n, 0)
                if cnt == 0:
                    return ["color: #444444"] * len(row)
                if n == 0:
                    return ["color: #00c853; font-weight: bold"] * len(row)
                if n <= 2:
                    return ["color: #52b788"] * len(row)
                if n <= 5:
                    return ["color: #f4a261"] * len(row)
                return ["color: #ef233c"] * len(row)

            st.dataframe(
                df_ws.style.apply(style_ws_rows, axis=1),
                use_container_width=True,
                hide_index=True,
                height=min(36 * (max_ws_seen + 2) + 38, 500),
            )

        # Capital vs Trade chart
        st.subheader("Capital per Trade")
        _cap = capital
        _trade_caps = [_cap]
        for _cycle in result.cycles:
            for _step in _cycle.steps:
                if _step.pnl != 0:
                    _cap = round(_cap + _step.pnl, 2)
                    _trade_caps.append(_cap)
        _colors = ["#00c853" if v >= capital else "#ef233c" for v in _trade_caps]
        fig_ct = go.Figure()
        fig_ct.add_hline(y=capital, line_dash="dash", line_color="#888", line_width=1)
        fig_ct.add_trace(go.Scatter(
            y=_trade_caps,
            mode="lines",
            line=dict(color="#f0a500", width=2),
            fill="tozeroy",
            fillcolor="rgba(240,165,0,0.08)",
            name="Capital (USD)",
        ))
        fig_ct.update_layout(
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa", xaxis_title="Trade #", yaxis_title="Capital (USD)",
        )
        fig_ct.update_xaxes(showgrid=False)
        fig_ct.update_yaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig_ct, use_container_width=True)

        # Equity curve (P&L delta from starting capital)
        st.subheader("Equity Curve (Cumulative P&L)")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=result.equity_curve,
            mode="lines",
            line=dict(color="#4cc9f0", width=2),
            fill="tozeroy",
            fillcolor="rgba(76,201,240,0.1)",
            name="Cumulative P&L",
        ))
        fig_eq.update_layout(
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa", xaxis_title="Cycle #", yaxis_title="Net P&L (USD)",
        )
        fig_eq.update_xaxes(showgrid=False)
        fig_eq.update_yaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Whipsaws per cycle bar chart
        st.subheader("Whipsaws per Cycle")
        fig_ws = go.Figure(go.Bar(
            y=[c.whipsaws for c in result.cycles],
            marker_color=[
                "#ef233c" if c.whipsaws > 5 else "#4cc9f0" if c.whipsaws <= 2 else "#f4a261"
                for c in result.cycles
            ],
            name="Whipsaws",
        ))
        fig_ws.update_layout(
            height=250, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa", xaxis_title="Cycle #", yaxis_title="Whipsaws",
        )
        fig_ws.update_xaxes(showgrid=False)
        fig_ws.update_yaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig_ws, use_container_width=True)

        # Cycle detail table
        st.subheader("Cycle Log")
        rows = []
        running_capital = capital
        running_capitals = []
        for c in result.cycles:
            running_capital += c.net_pnl
            running_capitals.append(round(running_capital, 2))

        if not running_capitals:
            st.info("No cycles completed with these settings — try a lower Reward Ratio or different buffer/leverage.")
            st.stop()
        min_capital = min(running_capitals)
        # Only highlight the lowest-capital row if it actually dipped below starting capital.
        min_capital_idx = running_capitals.index(min_capital) if min_capital < capital else -1

        for idx, c in enumerate(result.cycles):
            start_dt = datetime.datetime.utcfromtimestamp(c.start_ts / 1000).strftime("%Y-%m-%d %H:%M")
            end_dt = datetime.datetime.utcfromtimestamp(c.end_ts / 1000).strftime("%Y-%m-%d %H:%M")
            rows.append({
                "#": c.cycle_num,
                "Capital After": running_capitals[idx],
                "P&L": f"+${c.net_pnl:,.2f}" if c.net_pnl > 0 else (f"-${abs(c.net_pnl):,.2f}" if c.net_pnl < 0 else "$0.00"),
                "Entry": f"{c.entry_price:,.4f}",
                "ATR": f"{c.atr_at_entry:,.4f}",
                "LP": f"{c.lp:,.4f}",
                "SP": f"{c.sp:,.4f}",
                "Whipsaws": c.whipsaws,
                "Duration": f"{c.duration_candles} candles",
                "Worst Loss Before TP": c.peak_intra_loss,
                "Exit": "💀 LIQUIDATED" if not c.capital_ok else ("⛔ STOPPED" if c.exit_direction == "STOPPED" else c.exit_direction),
                "Start": start_dt,
                "End": end_dt,
            })

        df_cycles = pd.DataFrame(rows)

        def style_cycle_rows(row):
            idx = row.name
            cols = list(row.index)
            gl_col = cols.index("P&L") if "P&L" in cols else None
            if idx == min_capital_idx:
                styles = ["background-color: #4a1010; color: #ff8080"] * len(row)
            elif not result.cycles[idx].capital_ok:
                styles = ["background-color: #3d0000; color: #ff4444"] * len(row)
            elif result.cycles[idx].exit_direction == "STOPPED":
                styles = ["background-color: #2a1f00; color: #f4a261"] * len(row)
            else:
                styles = [""] * len(row)
            # Colour Gain/Loss cell green / red / grey
            if gl_col is not None:
                pnl = result.cycles[idx].net_pnl
                if pnl > 0:
                    styles[gl_col] = "color: #00c853; font-weight: bold"
                elif pnl < 0:
                    styles[gl_col] = "color: #ff4444; font-weight: bold"
                else:
                    styles[gl_col] = "color: #888888"
            return styles

        display_df = df_cycles.copy()
        display_df["Capital After"] = display_df["Capital After"].apply(lambda x: f"${x:,.2f}")
        display_df["Worst Loss Before TP"] = display_df["Worst Loss Before TP"].apply(lambda x: f"${x:,.2f}")

        drawdown_note = f" · Row highlighted in red = lowest capital point (${min_capital:,.2f})" if min_capital_idx >= 0 else ""
        st.caption(f"Starting capital: **${capital:,.2f}**{drawdown_note}")
        st.dataframe(
            display_df.style.apply(style_cycle_rows, axis=1),
            use_container_width=True, hide_index=True,
        )

        # Auto-save to server history + download button
        meta = {
            "symbol": symbol,
            "trading_tf": trading_tf if use_live else "manual",
            "atr_tf": atr_tf if use_live else "manual",
            "buffer": efficiency_buffer,
            "leverage": leverage,
            "base_lots": base_lots,
            "capital": capital,
            "lookback_days": bt_days,
            "total_cycles": result.total_cycles,
            "winning_cycles": result.winning_cycles,
            "total_net_pnl": result.total_net_pnl,
            "liquidated": result.liquidated,
            "saved_utc": datetime.datetime.utcnow().isoformat(),
        }
        saved_path = save_backtest_csv(df_cycles, symbol, meta)

        csv = df_cycles.to_csv(index=False)
        pdf_bytes = generate_backtest_pdf(
            result=result,
            symbol=symbol,
            tf_display=trading_tf if use_live else "manual",
            atr_display=atr_tf if use_live else "manual",
            atr_period=atr_period,
            efficiency_buffer=efficiency_buffer,
            reward_ratio=reward_ratio,
            base_lots=base_lots,
            leverage=leverage,
            capital=capital,
            bt_days=bt_days,
            fee_rate=fee_rate,
        )
        pdf_name = saved_path.stem + ".pdf"
        dl1, dl2 = st.columns([1, 1])
        with dl1:
            st.download_button(
                "⬇ Download Cycle Log (CSV)", csv,
                file_name=saved_path.name, mime="text/csv",
            )
        with dl2:
            st.download_button(
                "⬇ Download Summary (PDF)", pdf_bytes,
                file_name=pdf_name, mime="application/pdf",
            )


# ══════════════════════════════════════════════════════════════════════════════
# BOT CONFIG MODE
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "Bot Config":
    st.title(f"CHV Bot Config Generator — {symbol}")

    if not params.is_safe:
        st.error(f"Cannot generate config — ATR TRAP detected.\n\n{params.safety_message}")
        st.stop()

    st.subheader("Verify Parameters Before Generating")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("LP  Long Entry", f"{params.lp:,.4f}")
        st.metric("SP  Short Entry", f"{params.sp:,.4f}")
        st.metric("TL  Long TP", f"{params.tl:,.4f}")
        st.metric("TS  Short TP", f"{params.ts:,.4f}")
    with col2:
        st.metric("Cut Loss Zone (c)", f"{params.c:,.4f}")
        st.metric("Recovery Target (d)", f"{params.d:,.4f}")
        st.metric("Footprint", f"{params.footprint:,.4f}  ({params.footprint_pct*100:.1f}% of ATR)")
        st.markdown(status_badge(params.safety_status), unsafe_allow_html=True)

    exchange = st.selectbox("Target Exchange", ["binance", "bybit", "okx"], index=0)

    st.subheader("Exchange Settings")
    testnet = st.checkbox("Start in Testnet mode (recommended)", value=True)
    hedge_mode = st.checkbox("Enable Hedge Mode (required for CHV)", value=True)

    if st.button("Generate Bot Config", type="primary"):
        config = generate_bot_config(params, base_lots, leverage, capital, fee_rate, exchange)
        config["exchange"]["testnet"] = testnet
        config["execution"]["hedge_mode"] = hedge_mode

        config_json = json.dumps(config, indent=2)

        st.subheader("Bot Configuration JSON")
        st.caption("Copy this into your bot's config file. Fill in your API keys before going live.")
        st.code(config_json, language="json")

        st.download_button(
            label="Download config.json",
            data=config_json,
            file_name=f"chv_{symbol.lower()}_config.json",
            mime="application/json",
        )

        st.warning(
            "Important checklist before going live:\n"
            "1. Fill in your API key and secret\n"
            "2. Set `testnet: false` only after validating on testnet\n"
            "3. Set `strategy.active: true` to enable execution\n"
            "4. Ensure Hedge Mode is enabled in Binance Futures settings\n"
            "5. Verify LP/SP/TL/TS match your current chart levels"
        )


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY MODE
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "History":
    st.title("📂 Backtest History")
    st.caption("Every backtest run is auto-saved here. Download any file to study it offline.")

    files = sorted(BACKTEST_DIR.glob("chv_backtest_*.csv"), reverse=True)

    if not files:
        st.info("No backtests saved yet. Run a backtest first and it will appear here.")
    else:
        # Parse metadata from the first comment line of each file
        rows = []
        for f in files:
            try:
                with open(f) as fh:
                    first_line = fh.readline().strip()
                meta = json.loads(first_line.lstrip("# ")) if first_line.startswith("# {") else {}
            except Exception:
                meta = {}

            size_kb = f.stat().st_size / 1024
            rows.append({
                "File": f.name,
                "Symbol": meta.get("symbol", "—"),
                "Lookback": f"{meta.get('lookback_days', '—')}d",
                "TF": meta.get("trading_tf", "—"),
                "Capital": f"${meta.get('capital', 0):,.0f}" if meta.get("capital") else "—",
                "Leverage": f"{meta.get('leverage', '—')}×",
                "Cycles": meta.get("total_cycles", "—"),
                "Net PnL": f"${meta.get('total_net_pnl', 0):,.2f}" if meta.get("total_net_pnl") is not None else "—",
                "Liquidated": "💥 Yes" if meta.get("liquidated") else "✅ No",
                "Saved (UTC)": meta.get("saved_utc", "—")[:19].replace("T", " ") if meta.get("saved_utc") else "—",
                "Size": f"{size_kb:.1f} KB",
                "_path": str(f),
            })

        df_hist = pd.DataFrame(rows)

        # Symbol filter
        symbols_in_history = ["All"] + sorted(df_hist["Symbol"].unique().tolist())
        filter_sym = st.selectbox("Filter by symbol", symbols_in_history)
        if filter_sym != "All":
            df_hist = df_hist[df_hist["Symbol"] == filter_sym]

        st.dataframe(df_hist.drop(columns=["_path"]), use_container_width=True, hide_index=True)
        st.caption(f"**{len(df_hist)}** backtest(s) shown · {len(files)} total on server")

        st.divider()
        st.subheader("Download a file")
        selected_name = st.selectbox(
            "Select backtest to download",
            [r["File"] for r in rows if filter_sym == "All" or r["Symbol"] == filter_sym],
        )
        if selected_name:
            fpath = BACKTEST_DIR / selected_name
            with open(fpath, "rb") as fh:
                raw = fh.read()
            st.download_button(
                f"⬇ Download {selected_name}",
                data=raw,
                file_name=selected_name,
                mime="text/csv",
            )

        st.divider()
        st.subheader("🗑 Clear History")
        if st.button("Clear All CSV Files", type="secondary"):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning(f"This will permanently delete **{len(files)}** backtest file(s) from the server. Are you sure?")
            col_yes, col_no = st.columns([1, 5])
            with col_yes:
                if st.button("Yes, delete all", type="primary"):
                    deleted = 0
                    for f in files:
                        try:
                            f.unlink()
                            deleted += 1
                        except Exception:
                            pass
                    st.session_state["confirm_clear"] = False
                    st.success(f"Deleted {deleted} file(s).")
                    st.rerun()
            with col_no:
                if st.button("Cancel"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()
