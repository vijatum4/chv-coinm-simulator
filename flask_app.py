"""CHV CoinM Simulator — Flask App.

Runs on port 8507. Teal dark theme.
"""
from __future__ import annotations
import io, json, os, pathlib, sys, math, datetime, threading, uuid, csv
from collections import Counter, defaultdict
from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, Response, send_file, session)
from pathlib import Path

SIM_DIR = Path(__file__).parent
sys.path.insert(0, str(SIM_DIR))

from chv_engine import calculate_params, simulate, optimize, generate_bot_config, CHVParams, SimResult
from backtest_engine import run_backtest
from data_fetcher import (fetch_historical_ohlcv, klines_to_candles,
                          fetch_price_and_atr, fetch_price_and_atr_as_of,
                          get_coinm_symbols, coinm_contract_size, coinm_base_asset)

app = Flask(__name__, template_folder=str(SIM_DIR / 'templates'),
            static_folder=str(SIM_DIR / 'static'))
app.secret_key = 'chv-sim-flask-2025-x7k'

BACKTEST_DIR = SIM_DIR / 'backtests'
BACKTEST_DIR.mkdir(exist_ok=True)

_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()

# ── Constants ─────────────────────────────────────────────────────────────────
# Match Streamlit app.py options exactly
SLIP_LABELS = [
    '0.00% — No slippage (limit orders)',
    '0.05% — Liquid pairs (BTC / ETH)',
    '0.10% — Standard alts',
    '0.20% — Volatile / small coins',
]
SLIP_VALUES = [0.0, 0.0005, 0.001, 0.002]   # fractions, not percentages

BUFFER_CHOICES = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

TF_CHOICES = ['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d']

# Auto ATR TF rule (matches Streamlit TF_RULE)
TF_RULE = {
    '1m': '5m', '3m': '5m', '5m': '30m', '15m': '1h',
    '30m': '2h', '1h': '4h', '2h': '4h', '4h': '1d',
    '6h': '1d', '8h': '1d', '12h': '3d', '1d': '1w',
}

# CoinM contract specs: (min_contracts, step, decimals)
# All CoinM symbols use integer contracts (step=1), face_value from coinm_contract_size()
# All CoinM symbols: integer contracts, step=1, dec=0. BTC=$100/contract, others=$10.
LOT_SPECS = {s: (1, 1, 0) for s in [
    'AAVEUSD_PERP', 'ADAUSD_PERP', 'AVAXUSD_PERP', 'BCHUSD_PERP',
    'BNBUSD_PERP',  'BTCUSD_PERP', 'DOGEUSD_PERP', 'DOTUSD_PERP',
    'ETCUSD_PERP',  'ETHUSD_PERP', 'FILUSD_PERP',  'LINKUSD_PERP',
    'LTCUSD_PERP',  'NEARUSD_PERP','SOLUSD_PERP',  'SUIUSD_PERP',
    'TRXUSD_PERP',  'UNIUSD_PERP', 'XLMUSD_PERP',  'XRPUSD_PERP',
]}
MAX_STEPS = 50


# ── Template filters ─────────────────────────────────────────────────────────
@app.template_filter('capfmt')
def capfmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v >= 1000:
        k = v / 1000
        return f'{int(k)}k' if k == int(k) else f'{k:.1f}k'
    return f'{v:,.0f}'


@app.template_filter('btcfmt')
def btcfmt(v):
    """Format base-asset P&L with sign and appropriate decimal places."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    sign = '+' if v >= 0 else ''
    av = abs(v)
    if av >= 1:       return f'{sign}{v:.4f}'
    if av >= 0.001:   return f'{sign}{v:.5f}'
    if av >= 0.00001: return f'{sign}{v:.7f}'
    return f'{sign}{v:.8f}'


@app.template_filter('coinfmt')
def coinfmt(v):
    """Format base-asset amount (no sign) with appropriate decimal places."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '—'
    av = abs(v)
    if av >= 1:       return f'{v:.4f}'
    if av >= 0.001:   return f'{v:.5f}'
    if av >= 0.00001: return f'{v:.7f}'
    return f'{v:.8f}'


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sidebar_defaults():
    """Defaults that match the Streamlit app exactly."""
    return dict(
        symbol='BTCUSD_PERP',
        trading_tf='1h',
        atr_tf='4h',
        atr_period=5,
        price_val=0.0,
        atr_val=0.0,
        efficiency_buffer=0.80,
        reward_ratio=2.5,
        base_lots=1.0,       # contracts (integer)
        leverage=10,
        capital=0.01,        # BTC (base asset)
        fee_rate_pct=0.05,   # taker
        maker_fee_pct=0.02,  # maker (TP limit orders)
        slippage_label='0.05% — Liquid pairs (BTC / ETH)',
        ws_limit_on=False,
        ws_limit=7,
        atr_guard_on=True,
        atr_guard_multiplier=1.0,
        use_live=True,
        base_asset='BTC',
        face_value=100.0,    # $100/contract for BTC
        dual_atr_on=False,
        atr_period_2=14,
        dual_atr_mode='min',
        min_notional_on=False,
        fixed_margin_on=False,
        target_margin=50.0,
        sl_mode='wick',
    )


def _parse_sidebar(form, sess):
    d = _sidebar_defaults()
    src = {**sess, **form}

    sym = src.get('symbol', d['symbol'])
    d['symbol'] = sym
    d['base_asset'] = coinm_base_asset(sym)
    d['face_value'] = float(coinm_contract_size(sym))

    for key in ('trading_tf', 'atr_tf', 'dual_atr_mode', 'sl_mode'):
        if src.get(key):
            d[key] = src[key]

    # Auto-update atr_tf when trading_tf changes
    if form.get('trading_tf'):
        d['atr_tf'] = TF_RULE.get(form['trading_tf'], '4h')

    for key in ('atr_period', 'atr_period_2', 'leverage', 'ws_limit'):
        try:
            d[key] = int(src[key])
        except (KeyError, TypeError, ValueError):
            pass

    for key in ('price_val', 'atr_val', 'efficiency_buffer', 'reward_ratio',
                'base_lots', 'capital', 'fee_rate_pct', 'maker_fee_pct', 'atr_guard_multiplier', 'target_margin'):
        try:
            d[key] = float(src[key])
        except (KeyError, TypeError, ValueError):
            pass

    # Apply capital default when no explicit value was submitted
    if 'capital' not in form and 'capital' not in sess:
        d['capital'] = 0.01

    for key in ('ws_limit_on', 'atr_guard_on', 'use_live', 'dual_atr_on', 'min_notional_on', 'fixed_margin_on'):
        # When a form is submitted, absent checkbox = explicitly unchecked.
        # Only fall back to session on GET (form is empty {}).
        val = form.get(key, '') if form else src.get(key, '')
        d[key] = bool(val and val not in ('', 'off', 'false', '0', False))

    sl = src.get('slippage_label', d['slippage_label'])
    d['slippage_label'] = sl
    try:
        d['slip_pct'] = SLIP_VALUES[SLIP_LABELS.index(sl)]
    except (ValueError, IndexError):
        d['slip_pct'] = 0.0005  # default to liquid pairs

    # CoinM: all contracts use integer step (1 contract minimum)
    min_qty, step, dec = LOT_SPECS.get(sym, (1, 1, 0))
    d['lot_min'] = min_qty
    d['lot_step'] = step
    d['lot_dec'] = dec
    if d['base_lots'] < min_qty:
        d['base_lots'] = max(1, min_qty)

    return d


def _save_to_session(d):
    skip = {'slip_pct', 'base_asset', 'settle_coin', 'lot_min', 'lot_step', 'lot_dec', 'face_value'}
    for k, v in d.items():
        if k not in skip:
            session[k] = v


# ── SVG Chart helpers ─────────────────────────────────────────────────────────
def _build_bt_charts(result, capital_btc: float, pnl_btc_list: list,
                     base_asset: str = 'BTC',
                     ws_gold_thresh: int = 7, ws_red_thresh: int = 11) -> dict:
    """Build SVG path data for all backtest charts (values in base asset)."""
    cycles = result.cycles
    if not cycles:
        return {}
    n = len(cycles)
    x0, x1 = 40, 710

    def _lbl(v):
        av = abs(v)
        if av >= 1:       return f'{v:.3f}'
        if av >= 0.001:   return f'{v:.4f}'
        if av >= 0.00001: return f'{v:.6f}'
        return f'{v:.8f}'

    # ── Equity curve in BTC ───────────────────────────────────────────────
    eq_vals = [capital_btc]
    for i, c in enumerate(cycles):
        pnl_btc = pnl_btc_list[i] if i < len(pnl_btc_list) else 0.0
        eq_vals.append(eq_vals[-1] + pnl_btc)
    eq_min = min(eq_vals); eq_max = max(eq_vals)
    if eq_min == eq_max:
        eq_min -= 1; eq_max += 1
    eq_pad = (eq_max - eq_min) * 0.05
    eq_min -= eq_pad; eq_max += eq_pad
    eq_rng = eq_max - eq_min

    def py_eq(v): return round(240 + ((v - eq_min) / eq_rng) * (40 - 240), 1)
    eq_pts = [(round(x0 + i / n * (x1 - x0), 1), py_eq(v)) for i, v in enumerate(eq_vals)]
    eq_line = 'M ' + ' L '.join(f'{x},{y}' for x, y in eq_pts)
    eq_area = eq_line + f' L {eq_pts[-1][0]},240 L {eq_pts[0][0]},240 Z'
    eq_ep = eq_pts[-1]
    eq_zero_y = py_eq(capital_btc)

    def _y_lbls(ymin, ymax, steps=4, y_bot=240, y_top=40):
        rng = ymax - ymin or 1
        out = []
        for i in range(steps + 1):
            v = ymin + rng * i / steps
            yp = round(y_bot + ((v - ymin) / rng) * (y_top - y_bot), 1)
            out.append({'y': yp, 'label': _lbl(v)})
        return out

    def _x_lbls(total, steps=5):
        out = []
        for i in range(steps + 1):
            xp = round(x0 + i / steps * (x1 - x0), 1)
            out.append({'x': xp, 'label': str(int(total * i / steps))})
        return out

    # ── Capital per cycle ─────────────────────────────────────────────────
    cap_vals = list(eq_vals[1:])  # one point per cycle end
    cap_min = min(cap_vals); cap_max = max(cap_vals)
    if cap_min == cap_max:
        cap_min -= 1; cap_max += 1
    cap_pad = (cap_max - cap_min) * 0.05
    cap_min -= cap_pad; cap_max += cap_pad
    cap_rng = cap_max - cap_min

    def py_cap(v): return round(220 + ((v - cap_min) / cap_rng) * (20 - 220), 1)
    cap_pts = [(round(x0 + i / (n - 1 or 1) * (x1 - x0), 1), py_cap(v)) for i, v in enumerate(cap_vals)]
    cap_line = 'M ' + ' L '.join(f'{x},{y}' for x, y in cap_pts)
    cap_area = cap_line + f' L {cap_pts[-1][0]},220 L {cap_pts[0][0]},220 Z'

    # ── Capital per trade in BTC (every step with pnl != 0) ─────────────
    ct_vals = [capital_btc]
    for i, c in enumerate(cycles):
        pnl_btc = pnl_btc_list[i] if i < len(pnl_btc_list) else 0.0
        n_steps = len([s for s in c.steps if s.pnl != 0])
        if n_steps:
            step_pnl_btc = pnl_btc / n_steps  # distribute BTC P&L across steps
            for step in c.steps:
                if step.pnl != 0:
                    ct_vals.append(ct_vals[-1] + step_pnl_btc)
    m = len(ct_vals)
    ct_min = min(ct_vals); ct_max = max(ct_vals)
    if ct_min == ct_max:
        ct_min -= 1; ct_max += 1
    ct_pad = (ct_max - ct_min) * 0.05
    ct_min -= ct_pad; ct_max += ct_pad
    ct_rng = ct_max - ct_min

    def py_ct(v): return round(220 + ((v - ct_min) / ct_rng) * (20 - 220), 1)
    ct_pts = [(round(x0 + i / (m - 1 or 1) * (x1 - x0), 1), py_ct(v)) for i, v in enumerate(ct_vals)]
    ct_line = 'M ' + ' L '.join(f'{x},{y}' for x, y in ct_pts)
    ct_area = ct_line + f' L {ct_pts[-1][0]},220 L {ct_pts[0][0]},220 Z'

    # ── Whipsaws per cycle bars ───────────────────────────────────────────
    ws_vals = [c.whipsaws for c in cycles]
    ws_max_val = max(ws_vals) if ws_vals else 1
    bar_w = max(1.0, (x1 - x0) / n * 0.85)
    ws_bars_svg = ''
    for i, w in enumerate(ws_vals):
        bx = round(x0 + i / n * (x1 - x0), 1)
        bh = round((w / ws_max_val) * 180, 1) if ws_max_val else 0
        by = 220 - bh
        if w == 0:
            col = '#7A64EB'  # WS0 stays Solana Purple
        elif w >= ws_red_thresh:
            col = '#E63946'
        elif w >= ws_gold_thresh:
            col = '#ffc72e'
        else:
            col = '#39C3C4'
        ws_bars_svg += f'<rect x="{bx}" y="{by}" width="{bar_w:.1f}" height="{bh}" fill="{col}" rx="2"/>'

    return dict(
        # Equity curve — chart area y: 40 (top) … 240 (bottom)
        eq_line=eq_line, eq_area=eq_area, eq_ep_x=eq_ep[0], eq_ep_y=eq_ep[1],
        eq_zero_y=eq_zero_y,
        eq_y_lbls=_y_lbls(eq_min, eq_max, y_bot=240, y_top=40),
        eq_x_lbls=_x_lbls(n),
        eq_cycles=n,
        # Capital per cycle — chart area y: 20 (top) … 220 (bottom)
        cap_line=cap_line, cap_area=cap_area,
        cap_y_lbls=_y_lbls(cap_min, cap_max, y_bot=220, y_top=20),
        cap_x_lbls=_x_lbls(n),
        cap_cycles=n,
        # Capital per trade — chart area y: 20 (top) … 220 (bottom)
        ct_line=ct_line, ct_area=ct_area,
        ct_y_lbls=_y_lbls(ct_min, ct_max, y_bot=220, y_top=20),
        ct_x_lbls=_x_lbls(m),
        ct_trades=m - 1,
        # Whipsaws bars — chart area y: 40 (top) … 220 (bottom)
        ws_bars_svg=ws_bars_svg,
        ws_y_lbls=[{'y': round(220 - i / 4 * 180, 1), 'label': str(int(ws_max_val * i / 4))} for i in range(5)],
        ws_x_lbls=_x_lbls(n),
    )


# Keys match backtest_engine exit_direction values exactly
_EXIT_DISPLAY = {
    'EXIT_LONG':  'LONG',
    'EXIT_SHORT': 'SHORT',
    'STOPPED':    'Stopped',
    'ABORTED':    'Aborted',
    'LIQ':        'Liquidated',
}

def _prep_bt_log(result, capital_btc: float, pnl_btc_list: list,
                 trading_tf: str = '1h',
                 reward_ratio: float = 2.5, lot_dec: int = 3) -> list:
    cap_btc = capital_btc
    rows = []
    for i, c in enumerate(result.cycles):
        pnl_btc = pnl_btc_list[i] if i < len(pnl_btc_list) else 0.0
        cap_btc += pnl_btc
        s_dt = datetime.datetime.utcfromtimestamp(c.start_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
        e_dt = datetime.datetime.utcfromtimestamp(c.end_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.end_ts else '—'
        liq = not c.capital_ok
        aborted = c.exit_direction == 'ABORTED'
        exit_str = 'LIQ' if liq else ('ABORTED' if aborted else c.exit_direction)
        exit_display = _EXIT_DISPLAY.get(exit_str, exit_str.replace('_', ' '))
        c_param = c.lp - c.sp
        d_param = reward_ratio * c_param
        if exit_str == 'EXIT_LONG':
            tp_fmt = f'{c.lp + d_param:.4f}'
        elif exit_str == 'EXIT_SHORT':
            tp_fmt = f'{c.sp - d_param:.4f}'
        else:
            tp_fmt = '—'

        # BTC formatting
        def _bfmt(v):
            av = abs(v)
            s = '+' if v >= 0 else ''
            if av >= 1:       return f'{s}{v:.4f}'
            if av >= 0.001:   return f'{s}{v:.5f}'
            if av >= 0.00001: return f'{s}{v:.7f}'
            return f'{s}{v:.8f}'

        def _cfmt(v):
            av = abs(v)
            if av >= 1:       return f'{v:.4f}'
            if av >= 0.001:   return f'{v:.5f}'
            if av >= 0.00001: return f'{v:.7f}'
            return f'{v:.8f}'

        worst_btc = c.peak_intra_loss / c.entry_price

        rows.append({
            'num': c.cycle_num,
            'capital': _cfmt(cap_btc),
            'pnl': pnl_btc,
            'pnl_fmt': _bfmt(pnl_btc),
            'pnl_pos': pnl_btc > 0,
            'pnl_zero': pnl_btc == 0,
            'lots': f'{c.base_lots:.{lot_dec}f}',
            'entry': f'{c.entry_price:.4f}',
            'tp': tp_fmt,
            'atr': f'{c.atr_at_entry:.4f}',
            'lp': f'{c.lp:.4f}',
            'sp': f'{c.sp:.4f}',
            'ws': c.whipsaws,
            'ws_color': 'pos' if c.whipsaws == 0 else ('warn' if c.whipsaws <= 5 else 'neg'),
            'dur': f'{c.duration_candles} candles',
            'worst': (_bfmt(worst_btc) if worst_btc != 0 else '0'),
            'worst_neg': worst_btc < 0,
            'worst_zero': worst_btc == 0,
            'exit': exit_str,
            'exit_display': exit_display,
            'exit_long': 'LONG' in exit_str,
            'liq': liq,
            'aborted': aborted,
            'start': s_dt,
            'end': e_dt,
        })
    return rows


def _prep_trade_log(result) -> list:
    """Per-step trade history across all cycles. P&L in USD (template converts to asset)."""
    _ACTION = {
        'EXIT_LONG':  'TP Long',
        'EXIT_SHORT': 'TP Short',
        'STOPPED':    'Stopped',
    }
    rows = []
    trade_num = 0
    for cycle in result.cycles:
        for i, step in enumerate(cycle.steps):
            trade_num += 1
            # Action label
            if step.direction in _ACTION:
                action = _ACTION[step.direction]
                side_cls = 'long' if 'LONG' in step.direction else 'short'
            elif i == 0:
                action = f'Open {"Long" if step.direction == "LONG" else "Short"}'
                side_cls = step.direction.lower()
            else:
                action = f'→ {"Long" if step.direction == "LONG" else "Short"}'
                side_cls = step.direction.lower()

            ts = datetime.datetime.utcfromtimestamp(
                step.timestamp / 1000).strftime('%m-%d %H:%M') if step.timestamp else '—'

            rows.append({
                'num':      trade_num,
                'cycle':    cycle.cycle_num,
                'action':   action,
                'side_cls': side_cls,   # long | short
                'price':    f'{step.trigger_price:.4f}',
                'lots':     int(step.lots) if step.lots else '—',
                'pnl_usd':  step.pnl,   # USD — template converts ÷ entry_price
                'entry_price': cycle.entry_price,
                'ts':       ts,
            })
    return rows


def _build_ws_breakdown(result, ws_gold_thresh: int = 7, ws_red_thresh: int = 11) -> list:
    """Build whipsaw breakdown rows for the template."""
    all_ws = [c.whipsaws for c in result.cycles]
    if not all_ws:
        return []
    total = len(all_ws)
    counts = Counter(all_ws)
    max_ws = max(all_ws)
    max_cnt = max(counts.values())
    cumulative = 0
    rows = []
    for n in range(0, max_ws + 1):
        cnt = counts.get(n, 0)
        cumulative += cnt
        pct = cnt / total * 100
        cum_pct = cumulative / total * 100
        bar_w = (cnt / max_cnt * 100) if max_cnt else 0
        if n == 0:
            color = 'pos'          # WS0 stays Solana Purple
            hex_col = '#7A64EB'
        elif n >= ws_red_thresh:
            color = 'neg'
            hex_col = '#E63946'
        elif n >= ws_gold_thresh:
            color = 'warm'
            hex_col = '#ffc72e'
        else:
            color = 'safe'
            hex_col = '#39C3C4'
        rows.append({
            'n': n,
            'cnt': cnt,
            'pct': f'{pct:.1f}%',
            'cum': f'{cum_pct:.1f}%',
            'bar_w': round(bar_w, 1),
            'color': color,
            'hex_col': hex_col,
        })
    return rows


def _save_bt_csv(result, d: dict):
    meta = {
        'symbol': d['symbol'], 'trading_tf': d['trading_tf'],
        'atr_tf': d.get('atr_tf', '4h'), 'buffer': d['efficiency_buffer'],
        'leverage': d['leverage'], 'base_lots': d['base_lots'],
        'capital': d['capital'], 'lookback_days': d.get('bt_days', 365),
        'total_cycles': result.total_cycles,
        'winning_cycles': result.winning_cycles,
        'total_net_pnl': result.total_net_pnl,
        'liquidated': result.liquidated,
        'saved_utc': datetime.datetime.utcnow().isoformat(),
    }
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    fpath = BACKTEST_DIR / f"chv_backtest_{d['symbol']}_{ts}.csv"
    cap = d['capital']
    with open(fpath, 'w', newline='') as f:
        f.write('# ' + json.dumps(meta) + '\n')
        w = csv.writer(f)
        w.writerow(['#', 'Capital After', 'P&L', 'Entry', 'ATR', 'LP', 'SP',
                    'Whipsaws', 'Duration', 'Worst Loss', 'Exit', 'Start', 'End'])
        for c in result.cycles:
            cap += c.net_pnl
            sdt = datetime.datetime.utcfromtimestamp(c.start_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
            edt = datetime.datetime.utcfromtimestamp(c.end_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.end_ts else '—'
            exit_str = 'LIQ' if not c.capital_ok else getattr(c, 'exit_direction', '—')
            w.writerow([c.cycle_num, f'${cap:,.2f}',
                        f'+${c.net_pnl:,.2f}' if c.net_pnl >= 0 else f'-${abs(c.net_pnl):,.2f}',
                        f'{c.entry_price:.4f}', f'{c.atr_at_entry:.4f}',
                        f'{c.lp:.4f}', f'{c.sp:.4f}',
                        c.whipsaws, f'{c.duration_candles} candles',
                        f'${c.peak_intra_loss:,.2f}', exit_str, sdt, edt])


# ── Background backtest worker ────────────────────────────────────────────────
def _bt_worker(job_id: str):
    with _JOBS_LOCK:
        d = dict(_JOBS[job_id]['d'])

    def upd(p, msg):
        with _JOBS_LOCK:
            _JOBS[job_id].update(progress=p, msg=msg, status='running')

    try:
        trade_interval = d['trading_tf']
        atr_interval = d['atr_tf']
        bt_days = d.get('bt_days', 365)

        face_value = float(coinm_contract_size(d['symbol']))

        upd(5, f'Fetching {trade_interval} candles…')
        trading_klines, err1 = fetch_historical_ohlcv(
            d['symbol'], trade_interval, days=bt_days,
            market_type='coinm',
            progress_callback=lambda p: upd(int(5 + p * 35), f'Fetching {trade_interval} candles…'),
        )
        if err1:
            raise RuntimeError(f'Trading candles: {err1}')

        upd(40, f'Fetching {atr_interval} ATR candles…')
        atr_klines, err2 = fetch_historical_ohlcv(
            d['symbol'], atr_interval, days=bt_days,
            market_type='coinm',
            progress_callback=lambda p: upd(int(40 + p * 35), f'Fetching {atr_interval} candles…'),
        )
        if err2:
            raise RuntimeError(f'ATR candles: {err2}')

        trading_candles = klines_to_candles(trading_klines)
        atr_candles = klines_to_candles(atr_klines)

        # Derive BTC → USD conversion price from first available candle at backtest start
        start_candle_idx = int(d.get('atr_period', 5)) + 2
        conversion_price = trading_candles[min(start_candle_idx, len(trading_candles) - 1)].close
        capital_btc = float(d['capital'])    # user input is in base asset (BTC)
        capital_usd = capital_btc * conversion_price

        upd(78, 'Running CHV CoinM backtest…')
        result = run_backtest(
            symbol=d['symbol'],
            trading_candles=trading_candles,
            atr_candles=atr_candles,
            atr_period=int(d['atr_period']),
            atr_period_2=int(d['atr_period_2']) if d.get('dual_atr_on') else 0,
            dual_atr_mode=d.get('dual_atr_mode', 'min'),
            base_lots=d['base_lots'],
            leverage=int(d['leverage']),
            capital=capital_usd,
            fee_rate=d['fee_rate_pct'] / 100.0,
            fee_rate_maker=d.get('maker_fee_pct', 0.02) / 100.0,
            buffer=d['efficiency_buffer'],
            slippage_pct=d.get('slip_pct', 0.0005),
            reward_ratio=d['reward_ratio'],
            max_whipsaws=d['ws_limit'] if d['ws_limit_on'] else 0,
            atr_guard=d.get('atr_guard_on', True),
            atr_guard_multiplier=d.get('atr_guard_multiplier', 1.0),
            min_notional_on=bool(d.get('min_notional_on', False)),
            fixed_margin=float(d['target_margin']) if d.get('fixed_margin_on') else 0.0,
            lot_step=LOT_SPECS.get(d.get('symbol', ''), (1, 1, 0))[1],
            sl_mode=d.get('sl_mode', 'wick'),
            face_value=face_value,
        )
        upd(92, 'Saving CSV…')
        _save_bt_csv(result, d)

        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status='done', progress=100, msg='Done!',
                result=result, error=None,
                bt_capital=capital_usd,          # USD, used by engine
                bt_capital_btc=capital_btc,      # BTC input from user
                conversion_price=conversion_price,
            )
    except Exception as ex:
        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status='error', progress=0, msg=str(ex),
                result=None, error=str(ex),
            )


# ── Shared template context ────────────────────────────────────────────────────
def _ctx(d, mode):
    return dict(
        **d,
        mode=mode,
        slip_labels=SLIP_LABELS,
        buffer_choices=BUFFER_CHOICES,
        tf_choices=TF_CHOICES,
        tf_rule=TF_RULE,
        lot_specs=LOT_SPECS,
        available_symbols=get_coinm_symbols(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def root():
    return redirect('/sim/')


@app.route('/sim')
@app.route('/sim/')
def sim_root():
    return redirect(url_for('guide'))


@app.route('/sim/guide')
def guide():
    d = _parse_sidebar({}, session)
    return render_template('simulator/guide.html', **_ctx(d, 'guide'))


@app.route('/sim/about')
def about():
    d = _parse_sidebar({}, session)
    return render_template('simulator/about.html', **_ctx(d, 'about'))


@app.route('/sim/simulator', methods=['GET', 'POST'])
def simulator():
    d = _parse_sidebar(request.form if request.method == 'POST' else {}, session)
    _save_to_session(d)
    result = None
    error = None
    params = None
    if request.method == 'POST' and d['price_val'] > 0 and d['atr_val'] > 0:
        try:
            params = calculate_params(
                d['symbol'], d['price_val'], d['atr_val'],
                d['efficiency_buffer'], d['reward_ratio'],
                d['atr_guard_multiplier'],
            )
            result = simulate(
                params,
                d['base_lots'], d['leverage'], d['capital'],
                MAX_STEPS,
                d['fee_rate_pct'] / 100.0,
                face_value=d.get('face_value', 100.0),
            )
        except Exception as ex:
            error = str(ex)
    return render_template('simulator/simulator.html',
                           result=result, error=error, params=params,
                           **_ctx(d, 'simulator'))


@app.route('/sim/backtest', methods=['GET', 'POST'])
def backtest():
    d = _parse_sidebar(request.form if request.method == 'POST' else {}, session)
    _save_to_session(d)
    bt_job_id = None
    bt_error = None

    if request.method == 'POST' and request.form.get('action') == 'run':
        try:
            bt_days = int(request.form.get('bt_days', 365))
            d['bt_days'] = bt_days
            job_id = str(uuid.uuid4())[:8]
            with _JOBS_LOCK:
                _JOBS[job_id] = {
                    'status': 'queued', 'progress': 0, 'msg': 'Queued…',
                    'result': None, 'error': None,
                    'd': dict(d, bt_days=bt_days),
                }
            threading.Thread(target=_bt_worker, args=(job_id,), daemon=True).start()
            bt_job_id = job_id
        except Exception as ex:
            bt_error = str(ex)

    d.setdefault('bt_days', 365)
    return render_template('simulator/backtest.html',
                           bt_job_id=bt_job_id, bt_error=bt_error,
                           result=None, bt_log=[], bt_charts={},
                           total_trades=0, top6_ws=[], ws_breakdown=[],
                           **_ctx(d, 'backtest'))


@app.route('/sim/backtest/status/<job_id>')
def bt_status(job_id):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id, {}))
    return jsonify({
        'status': job.get('status', 'unknown'),
        'progress': job.get('progress', 0),
        'msg': job.get('msg', ''),
    })


@app.route('/sim/backtest/result/<job_id>')
def bt_result_view(job_id):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id, {}))
    if job.get('status') != 'done':
        return redirect(url_for('backtest'))

    result = job['result']
    d_job = job.get('d', {})
    bt_cap = job.get('bt_capital', 700.0)                  # USD (engine used this)
    capital_btc = job.get('bt_capital_btc', 0.01)          # BTC (user input)
    conversion_price = job.get('conversion_price', 70000.0)  # BTC/USD at backtest start

    d = _parse_sidebar({}, session)
    d.update({k: v for k, v in d_job.items()
              if k not in ('base_asset', 'settle_coin', 'lot_min', 'lot_step', 'lot_dec')})

    base_asset = coinm_base_asset(d_job.get('symbol', 'BTCUSD_PERP'))

    # ── BTC P&L per cycle (each cycle's USD P&L ÷ its entry price) ───────
    pnl_btc_list = [c.net_pnl / c.entry_price for c in result.cycles]
    total_pnl_btc = sum(pnl_btc_list)
    fees_btc = result.total_fees / conversion_price
    total_equity_btc = capital_btc + total_pnl_btc
    worst_intra_btc = result.worst_intra_loss / conversion_price
    max_drawdown_btc = result.max_drawdown / conversion_price

    # Running BTC capital for cap_at_worst
    cap_btc_running = capital_btc
    cap_btc_at_worst = capital_btc
    for i, c in enumerate(result.cycles):
        if c.cycle_num == result.worst_intra_loss_cycle:
            cap_btc_at_worst = cap_btc_running
        cap_btc_running += pnl_btc_list[i]

    total_trades = sum(len(c.steps) for c in result.cycles)
    growth_pct = (total_pnl_btc / capital_btc * 100) if capital_btc else 0
    _leverage   = float(d.get('leverage', 10))
    _base_lots  = float(d.get('base_lots', 1.0))
    _lot_dec    = LOT_SPECS.get(d.get('symbol', ''), (1, 1, 0))[2]

    # ── Liquidation detail ─────────────────────────────────────────────────
    liq_detail = None
    if result.liquidated:
        liq_c = next((c for c in result.cycles if not c.capital_ok), None)
        if liq_c:
            ws_n = result.liquidation_step  # whipsaw count at failure
            last_dir_step = next(
                (s for s in reversed(liq_c.steps) if s.direction in ('LONG', 'SHORT')), None
            )
            last_lots = last_dir_step.lots if last_dir_step else liq_c.base_lots
            last_dir  = last_dir_step.direction if last_dir_step else 'LONG'
            # SL price of the position that couldn't invert
            sl_price  = liq_c.sp if last_dir == 'LONG' else liq_c.lp
            # Lot size that would have been needed (but couldn't be opened)
            failed_lots = (round(liq_c.base_lots * 0.5, 6)
                           if ws_n == 1
                           else round(last_lots * 1.5, 6))
            _face_value = float(d_job.get('face_value', coinm_contract_size(d_job.get('symbol', 'BTCUSD_PERP'))))
            # CoinM: margin = contracts × face_value / leverage (price-independent)
            failed_margin = (failed_lots * _face_value) / _leverage
            # Capital at the START of the liquidation cycle
            cap_before = bt_cap
            for c in result.cycles:
                if c.cycle_num == liq_c.cycle_num:
                    break
                cap_before += c.net_pnl
            shortfall = failed_margin - cap_before
            cum_loss  = result.liquidation_loss          # negative number
            # Determine which engine condition actually fired:
            # A) margin_needed > capital  →  shortfall > 0
            # B) abs(balance) > capital   →  abs(cum_loss) > cap_before
            reason = 'margin' if shortfall > 0 else 'loss'
            liq_detail = {
                'cycle':         result.liquidation_cycle,
                'ws':            ws_n,
                'last_lots':     last_lots,
                'failed_lots':   failed_lots,
                'sl_price':      sl_price,
                'failed_margin': failed_margin,
                'cap_before':    cap_before,
                'shortfall':     shortfall,
                'cum_loss':      cum_loss,
                'net_remaining': cap_before + cum_loss,  # negative when losses > capital
                'reason':        reason,
            }

    # ── Min notional detail ────────────────────────────────────────────────
    mn_detail = None
    if result.min_notional_rejected:
        step = LOT_SPECS.get(d.get('symbol', ''), (1, 1, 0))[1]
        inv_lots_val = round(_base_lots * 0.5, 6)
        inv_lots_fmt = f'{inv_lots_val:.{_lot_dec}f}'
        # CoinM: WS1 needs ≥ 1 contract → base must be ≥ 2 contracts
        min_base = max(2, math.ceil(1.0 / max(inv_lots_val, 0.0001) / step) * step)
        min_base_fmt = f'{min_base:.{_lot_dec}f}'
        mn_detail = {
            'cycle':       result.min_notional_cycle,
            'entry_price': result.min_notional_price,
            'inv_lots':    inv_lots_fmt,
            'inv_notional': result.min_notional_inv_notional,
            'min_base':    min_base_fmt,
            'completed':   result.total_cycles,
        }

    _face_val_ws = float(d.get('face_value', coinm_contract_size(d.get('symbol', 'BTCUSD_PERP'))))

    def _ws_peak_info(cycle):
        N = cycle.whipsaws
        cb = cycle.base_lots
        lots = cb if N == 0 else cb * 0.5 * (1.5 ** max(N - 1, 0))
        # CoinM: margin = contracts × face_value / leverage (price-independent)
        margin = lots * _face_val_ws / _leverage
        lots_fmt = f'{lots:.{_lot_dec}f}'
        return lots_fmt, margin

    # Sort by (whipsaws DESC, margin DESC) — among ties use the higher-margin cycle
    top_cycles = sorted(
        result.cycles,
        key=lambda c: (c.whipsaws, _ws_peak_info(c)[1]),
        reverse=True,
    )[:5]
    top6_ws = [(c.whipsaws, *_ws_peak_info(c)) for c in top_cycles]
    # top6_ws[i] = (ws_count, lots_str, peak_margin)
    ws_breakdown = _build_ws_breakdown(result, ws_gold_thresh=ws_gold_thresh, ws_red_thresh=ws_red_thresh)

    # ── Severity thresholds ───────────────────────────────────────────────
    # Max Drawdown / Worst Cycle Loss: as % of starting capital
    drawdown_pct = (abs(max_drawdown_btc) / capital_btc * 100) if capital_btc else 0
    worst_pct    = (abs(worst_intra_btc) / cap_btc_at_worst * 100) if cap_btc_at_worst else 0

    # Max WS capacity: how many WS can starting capital theoretically fund?
    _face_val_cap = float(d_job.get('face_value',
                          coinm_contract_size(d_job.get('symbol', 'BTCUSD_PERP'))))
    _base_l = float(d_job.get('base_lots', 1))
    _inv = math.floor(_base_l * 0.5)
    _max_theo_ws = 0
    while _inv > 0:
        if (_inv * _face_val_cap) / _leverage > bt_cap:
            break
        _max_theo_ws += 1
        _inv = math.floor(_inv * 1.5)
    ws_red_thresh  = math.floor(_max_theo_ws * 0.85)
    ws_gold_thresh = math.floor(_max_theo_ws * 0.50)

    trade_log = _prep_trade_log(result)
    log = _prep_bt_log(result, capital_btc, pnl_btc_list,
                       d_job.get('trading_tf', '1h'),
                       reward_ratio=float(d_job.get('reward_ratio', 2.5)),
                       lot_dec=_lot_dec)
    charts = _build_bt_charts(result, capital_btc, pnl_btc_list, base_asset,
                              ws_gold_thresh=ws_gold_thresh, ws_red_thresh=ws_red_thresh)

    return render_template('simulator/backtest.html',
                           bt_job_id=job_id,
                           bt_error=None,
                           result=result,
                           bt_log=log,
                           trade_log=trade_log,
                           bt_charts=charts,
                           bt_cap=bt_cap,
                           capital_btc=capital_btc,
                           conversion_price=conversion_price,
                           total_pnl_btc=total_pnl_btc,
                           fees_btc=fees_btc,
                           total_equity_btc=total_equity_btc,
                           worst_intra_btc=worst_intra_btc,
                           max_drawdown_btc=max_drawdown_btc,
                           drawdown_pct=round(drawdown_pct, 1),
                           worst_pct=round(worst_pct, 1),
                           ws_red_thresh=ws_red_thresh,
                           ws_gold_thresh=ws_gold_thresh,
                           max_theoretical_ws=_max_theo_ws,
                           cap_btc_at_worst=cap_btc_at_worst,
                           total_trades=total_trades,
                           growth_pct=growth_pct,
                           top6_ws=top6_ws,
                           ws_breakdown=ws_breakdown,
                           liq_detail=liq_detail,
                           mn_detail=mn_detail,
                           currency_sym=base_asset,
                           **_ctx(d, 'backtest'))


@app.route('/sim/backtest/download-csv/<job_id>')
def bt_download_csv(job_id):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id, {}))
    if not job or job.get('status') != 'done':
        return 'Result not found', 404

    result = job['result']
    d = job.get('d', {})
    bt_cap = job.get('bt_capital', d.get('capital', 1000.0))

    buf = io.StringIO()
    meta = {'symbol': d.get('symbol'), 'exported_utc': datetime.datetime.utcnow().isoformat()}
    buf.write('# ' + json.dumps(meta) + '\n')
    w = csv.writer(buf)
    w.writerow(['#', 'Capital After', 'P&L', 'Entry', 'ATR', 'LP', 'SP',
                'Whipsaws', 'Duration', 'Worst Loss', 'Exit', 'Start', 'End'])
    cap = bt_cap
    for c in result.cycles:
        cap += c.net_pnl
        sdt = datetime.datetime.utcfromtimestamp(c.start_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
        edt = datetime.datetime.utcfromtimestamp(c.end_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.end_ts else '—'
        w.writerow([
            c.cycle_num, f'${cap:,.2f}',
            f'+${c.net_pnl:,.2f}' if c.net_pnl >= 0 else f'-${abs(c.net_pnl):,.2f}',
            f'{c.entry_price:.4f}', f'{c.atr_at_entry:.4f}',
            f'{c.lp:.4f}', f'{c.sp:.4f}',
            c.whipsaws, f'{c.duration_candles} candles',
            f'${c.peak_intra_loss:,.2f}',
            getattr(c, 'exit_direction', '—'), sdt, edt,
        ])
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    name = f"chv_backtest_{d.get('symbol', 'unknown')}_{ts}.csv"
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})


@app.route('/sim/optimizer', methods=['GET', 'POST'])
def optimizer_view():
    d = _parse_sidebar(request.form if request.method == 'POST' else {}, session)
    _save_to_session(d)

    opt_result = None
    opt_error = None
    opt_rows = []

    if request.method == 'POST' and request.form.get('action') == 'optimize':
        try:
            buffers_raw = request.form.getlist('opt_buffers') or ['0.75', '0.80', '0.85', '0.90']
            levs_raw = request.form.getlist('opt_leverages') or ['5', '10', '20', '25']
            lots_raw = request.form.get('opt_lots', '').strip()
            target_ws = int(request.form.get('target_ws', 5))

            buffers = [float(x) for x in buffers_raw]
            leverages = [int(x) for x in levs_raw]
            if lots_raw:
                lot_sizes = [float(x.strip()) for x in lots_raw.split(',') if x.strip()]
            else:
                lot_sizes = [1.0, 5.0, 10.0]

            if d['price_val'] <= 0 or d['atr_val'] <= 0:
                raise ValueError('Fetch price + ATR first before running optimizer.')

            sweep_results = optimize(
                symbol=d['symbol'],
                price=d['price_val'],
                atr=d['atr_val'],
                capital=d['capital'],
                fee_rate=d['fee_rate_pct'] / 100.0,
                buffers=buffers,
                leverages=leverages,
                lot_sizes=lot_sizes,
                max_steps=MAX_STEPS,
                face_value=d.get('face_value', 100.0),
            )

            for r in sweep_results:
                opt_rows.append({
                    'buffer': r.buffer,
                    'leverage': r.leverage,
                    'base_lots': r.base_lots,
                    'net_pnl': f'${r.net_pnl:,.2f}',
                    'steps_covered': r.steps_covered,
                    'capital_safe': r.capital_survives,
                    'footprint_pct': f'{r.footprint_pct * 100:.1f}%',
                    'status': r.safety_status,
                    'score': round(r.score, 4),
                })
            opt_rows.sort(key=lambda x: x['score'], reverse=True)
            opt_rows = opt_rows[:25]

        except Exception as ex:
            opt_error = str(ex)

    return render_template('simulator/optimizer.html',
                           opt_rows=opt_rows, opt_error=opt_error,
                           **_ctx(d, 'optimizer'))


@app.route('/sim/botconfig', methods=['GET', 'POST'])
def botconfig():
    d = _parse_sidebar(request.form if request.method == 'POST' else {}, session)
    _save_to_session(d)

    config_json = None
    bc_error = None
    params = None

    if d['price_val'] > 0 and d['atr_val'] > 0:
        try:
            params = calculate_params(
                d['symbol'], d['price_val'], d['atr_val'],
                d['efficiency_buffer'], d['reward_ratio'],
                d['atr_guard_multiplier'],
            )
        except Exception as ex:
            bc_error = str(ex)

    if request.method == 'POST' and params and params.is_safe:
        try:
            config = generate_bot_config(
                params, d['base_lots'], d['leverage'],
                d['capital'], d['fee_rate_pct'] / 100.0, 'binance',
            )
            config_json = json.dumps(config, indent=2)
        except Exception as ex:
            bc_error = str(ex)

    return render_template('simulator/botconfig.html',
                           params=params, config_json=config_json, bc_error=bc_error,
                           **_ctx(d, 'botconfig'))


@app.route('/sim/history')
def history():
    files = sorted(BACKTEST_DIR.glob('chv_backtest_*.csv'), reverse=True)
    entries = []
    for f in files:
        try:
            with open(f) as fp:
                meta_line = fp.readline()
            meta = json.loads(meta_line.lstrip('#').strip())
        except Exception:
            meta = {}
        entries.append({
            'name': f.name,
            'size_kb': round(f.stat().st_size / 1024, 1),
            'symbol': meta.get('symbol', '—'),
            'lookback': f"{meta.get('lookback_days', '—')}d",
            'trading_tf': meta.get('trading_tf', '—'),
            'capital': f"${meta.get('capital', 0):,.0f}" if meta.get('capital') else '—',
            'leverage': f"{meta.get('leverage', '—')}×",
            'cycles': meta.get('total_cycles', '—'),
            'net_pnl': (f"${meta.get('total_net_pnl', 0):,.2f}"
                        if meta.get('total_net_pnl') is not None else '—'),
            'liquidated': meta.get('liquidated', False),
            'saved': meta.get('saved_utc', '—')[:19].replace('T', ' ') if meta.get('saved_utc') else '—',
        })
    d = _parse_sidebar({}, session)
    return render_template('simulator/history.html',
                           entries=entries, **_ctx(d, 'history'))


@app.route('/sim/history/download/<path:filename>')
def history_download(filename):
    fpath = BACKTEST_DIR / filename
    if not fpath.exists():
        return 'Not found', 404
    return send_file(str(fpath), as_attachment=True, download_name=filename)


@app.route('/sim/history/delete', methods=['POST'])
def history_delete():
    fn = request.form.get('filename', '')
    fp = BACKTEST_DIR / fn
    if fp.exists() and fp.suffix == '.csv':
        fp.unlink()
    return redirect(url_for('history'))


@app.route('/sim/history/clear', methods=['POST'])
def history_clear():
    for f in BACKTEST_DIR.glob('chv_backtest_*.csv'):
        f.unlink()
    return redirect(url_for('history'))


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.route('/sim/api/symbols')
def api_symbols():
    return jsonify(get_coinm_symbols())


@app.route('/sim/api/price-atr')
def api_price_atr():
    sym           = request.args.get('symbol', 'BTCUSD_PERP')
    atr_tf        = request.args.get('atr_tf', '4h')
    period        = int(request.args.get('atr_period', 5))
    period_2      = int(request.args.get('atr_period_2', 0))
    bt_days       = int(request.args.get('bt_days', 0))
    dual_atr_mode = request.args.get('dual_atr_mode', 'min')
    pick          = min if dual_atr_mode == 'min' else max
    try:
        if bt_days > 0:
            import time as _time
            raw_ts   = int(_time.time() * 1000) - bt_days * 86_400_000
            midnight = (raw_ts // 86_400_000) * 86_400_000
            price, atr1, err, atr2 = fetch_price_and_atr_as_of(
                sym, atr_tf, period, as_of_ts=midnight, atr_period_2=period_2,
                market_type='coinm')
            if err:
                return jsonify({'ok': False, 'error': err})
            as_of_date = datetime.datetime.utcfromtimestamp(
                midnight / 1000).strftime('%Y-%m-%d')
            if period_2 > 0 and atr2:
                return jsonify({'ok': True, 'price': price,
                                'atr': round(pick(atr1, atr2), 6),
                                'atr1': atr1, 'atr2': atr2,
                                'as_of_date': as_of_date})
            return jsonify({'ok': True, 'price': price, 'atr': atr1,
                            'as_of_date': as_of_date})
        # ── live fetch ──────────────────────────────────────────────────────
        price, atr1, err = fetch_price_and_atr(sym, atr_tf, period, market_type='coinm')
        if err:
            return jsonify({'ok': False, 'error': err})
        if period_2 > 0:
            _, atr2, _ = fetch_price_and_atr(sym, atr_tf, period_2, market_type='coinm')
            atr_used = pick(atr1, atr2) if atr2 else atr1
            return jsonify({'ok': True, 'price': price,
                            'atr': round(atr_used, 6),
                            'atr1': atr1, 'atr2': atr2})
        return jsonify({'ok': True, 'price': price, 'atr': atr1})
    except Exception as ex:
        return jsonify({'ok': False, 'error': str(ex)})


@app.route('/sim/sw.js')
def service_worker():
    return Response('// CHV Sim SW', mimetype='application/javascript')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8507, debug=False, threaded=True)
