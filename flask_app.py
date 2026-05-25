"""CHV Recovery Trade Simulator — Flask App.

Runs on port 8503. Olive dark theme.
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
                          fetch_price_and_atr, get_available_symbols,
                          get_coinm_symbols)

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

# Per-symbol lot specs (min_qty, step, decimals) — matches Streamlit _LOT_SPECS
LOT_SPECS = {
    'AVAXUSDT':     (1.0,   1.0,   0),
    'BNBUSDT':      (0.01,  0.01,  2),
    'BTCUSDT':      (0.001, 0.001, 3),
    'DOGEUSDT':     (1.0,   1.0,   0),
    'ETHUSDT':      (0.001, 0.001, 3),
    'FARTCOINUSDT': (0.1,   0.1,   1),
    'JTOUSDT':      (1.0,   1.0,   0),
    'JUPUSDT':      (1.0,   1.0,   0),
    'NEARUSDT':     (1.0,   1.0,   0),
    'ONDOUSDT':     (0.1,   0.1,   1),
    'SOLUSDT':      (0.01,  0.01,  2),
    'SUIUSDT':      (0.1,   0.1,   1),
    'XAUUSDT':      (0.001, 0.001, 3),
    'XRPUSDT':      (0.1,   0.1,   1),
}
MAX_STEPS = 50  # matches Streamlit

# CoinM lot specs (lots = contracts; BTC=$100/contract, others=$10/contract)
COINM_LOT_SPECS = {
    'BTCUSD_PERP': (1, 1, 0),
    'ETHUSD_PERP': (1, 1, 0),
    'BNBUSD_PERP': (1, 1, 0),
    'XRPUSD_PERP': (1, 1, 0),
    'ADAUSD_PERP': (1, 1, 0),
    'SOLUSD_PERP': (1, 1, 0),
    'DOTUSD_PERP': (1, 1, 0),
    'LINKUSD_PERP': (1, 1, 0),
    'LTCUSD_PERP': (1, 1, 0),
    'BCHUSD_PERP': (1, 1, 0),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sidebar_defaults():
    """Defaults that match the Streamlit app exactly."""
    return dict(
        symbol='SOLUSDT',
        trading_tf='1h',
        atr_tf='4h',        # auto from TF_RULE for 1h
        atr_period=5,       # matches Streamlit default
        price_val=0.0,
        atr_val=0.0,
        efficiency_buffer=0.80,   # matches Streamlit default
        reward_ratio=2.5,
        base_lots=1.0,
        leverage=10,              # matches Streamlit default
        capital=1000.0,
        fee_rate_pct=0.05,
        slippage_label='0.05% — Liquid pairs (BTC / ETH)',  # matches Streamlit default index=1
        ws_limit_on=False,
        ws_limit=7,               # matches Streamlit default
        atr_guard_on=True,        # matches Streamlit default
        atr_guard_multiplier=1.0, # matches Streamlit default
        use_live=True,
        base_asset='SOL',
        settle_coin='USDT',
        market_type='USDM',
    )


def _parse_sidebar(form, sess):
    d = _sidebar_defaults()
    src = {**sess, **form}

    # Market type (USDM or COINM)
    d['market_type'] = form.get('market_type', src.get('market_type', 'USDM'))

    sym = src.get('symbol', d['symbol'])
    d['symbol'] = sym
    if '_PERP' in sym:
        d['base_asset'] = sym.split('USD')[0]
        d['settle_coin'] = 'USD'
    else:
        d['base_asset'] = sym.replace('USDT', '').replace('BUSD', '')
        d['settle_coin'] = 'USDT'

    for key in ('trading_tf', 'atr_tf'):
        if src.get(key):
            d[key] = src[key]

    # Auto-update atr_tf when trading_tf changes
    if form.get('trading_tf'):
        d['atr_tf'] = TF_RULE.get(form['trading_tf'], '4h')

    for key in ('atr_period', 'leverage', 'ws_limit'):
        try:
            d[key] = int(src[key])
        except (KeyError, TypeError, ValueError):
            pass

    for key in ('price_val', 'atr_val', 'efficiency_buffer', 'reward_ratio',
                'base_lots', 'capital', 'fee_rate_pct', 'atr_guard_multiplier'):
        try:
            d[key] = float(src[key])
        except (KeyError, TypeError, ValueError):
            pass

    for key in ('ws_limit_on', 'atr_guard_on', 'use_live'):
        val = src.get(key, '')
        d[key] = bool(val and val not in ('', 'off', 'false', '0', False))

    sl = src.get('slippage_label', d['slippage_label'])
    d['slippage_label'] = sl
    try:
        d['slip_pct'] = SLIP_VALUES[SLIP_LABELS.index(sl)]
    except (ValueError, IndexError):
        d['slip_pct'] = 0.0005  # default to liquid pairs

    # Lot spec for display
    if d.get('market_type') == 'COINM':
        min_qty, step, dec = COINM_LOT_SPECS.get(sym, (1, 1, 0))
    else:
        min_qty, step, dec = LOT_SPECS.get(sym, (0.001, 0.001, 3))
    d['lot_min'] = min_qty
    d['lot_step'] = step
    d['lot_dec'] = dec
    if d['base_lots'] < min_qty:
        d['base_lots'] = max(1.0, min_qty)

    return d


def _save_to_session(d):
    skip = {'slip_pct', 'base_asset', 'settle_coin', 'lot_min', 'lot_step', 'lot_dec'}
    for k, v in d.items():
        if k not in skip:
            session[k] = v


# ── SVG Chart helpers ─────────────────────────────────────────────────────────
def _build_bt_charts(result, capital: float) -> dict:
    """Build SVG path data for all backtest charts."""
    cycles = result.cycles
    if not cycles:
        return {}
    n = len(cycles)
    x0, x1 = 40, 710

    # ── Equity curve (cumulative capital after each cycle) ────────────────
    eq_vals = [capital]
    for c in cycles:
        eq_vals.append(eq_vals[-1] + c.net_pnl)
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
    eq_zero_y = py_eq(capital)

    def _y_lbls(ymin, ymax, steps=4):
        rng = ymax - ymin or 1
        out = []
        for i in range(steps + 1):
            v = ymin + rng * i / steps
            yp = round(240 + ((v - ymin) / rng) * (40 - 240), 1)
            if abs(v) >= 1000:
                lbl = f'${v/1000:.1f}k'
            elif abs(v) >= 10:
                lbl = f'${v:.0f}'
            else:
                lbl = f'${v:.1f}'
            out.append({'y': yp, 'label': lbl})
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

    # ── Capital per trade (every step with pnl != 0) ─────────────────────
    ct_vals = [capital]
    for c in cycles:
        for step in c.steps:
            if step.pnl != 0:
                ct_vals.append(ct_vals[-1] + step.pnl)
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
        col = '#B5E000' if w <= 2 else ('#FADD56' if w <= 5 else 'oklch(0.66 0.18 28)')
        ws_bars_svg += f'<rect x="{bx}" y="{by}" width="{bar_w:.1f}" height="{bh}" fill="{col}" rx="2"/>'

    return dict(
        # Equity curve
        eq_line=eq_line, eq_area=eq_area, eq_ep_x=eq_ep[0], eq_ep_y=eq_ep[1],
        eq_zero_y=eq_zero_y,
        eq_y_lbls=_y_lbls(eq_min, eq_max),
        eq_x_lbls=_x_lbls(n),
        eq_cycles=n,
        # Capital per cycle
        cap_line=cap_line, cap_area=cap_area,
        cap_y_lbls=_y_lbls(cap_min, cap_max),
        cap_x_lbls=_x_lbls(n),
        cap_cycles=n,
        # Capital per trade
        ct_line=ct_line, ct_area=ct_area,
        ct_y_lbls=_y_lbls(ct_min, ct_max),
        ct_x_lbls=_x_lbls(m),
        ct_trades=m - 1,
        # Whipsaws bars
        ws_bars_svg=ws_bars_svg,
        ws_y_lbls=[{'y': round(220 - i / 4 * 180, 1), 'label': str(int(ws_max_val * i / 4))} for i in range(5)],
        ws_x_lbls=_x_lbls(n),
    )


def _prep_bt_log(result, capital: float, trading_tf: str) -> list:
    TF_MS = {
        '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
        '1h': 3600000, '4h': 14400000, '12h': 43200000, '1d': 86400000,
    }
    cap = capital
    rows = []
    for c in result.cycles:
        cap += c.net_pnl
        s_dt = datetime.datetime.utcfromtimestamp(c.start_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
        e_dt = datetime.datetime.utcfromtimestamp(c.end_ts / 1000).strftime('%Y-%m-%d %H:%M') if c.end_ts else '—'
        liq = not c.capital_ok
        aborted = c.exit_direction == 'ABORTED'
        exit_str = 'LIQ' if liq else ('ABORTED' if aborted else c.exit_direction)
        rows.append({
            'num': c.cycle_num,
            'capital': f'${cap:,.2f}',
            'pnl': c.net_pnl,
            'pnl_fmt': (f'+${c.net_pnl:,.2f}' if c.net_pnl >= 0 else f'-${abs(c.net_pnl):,.2f}'),
            'pnl_pos': c.net_pnl >= 0,
            'entry': f'{c.entry_price:.4f}',
            'atr': f'{c.atr_at_entry:.4f}',
            'lp': f'{c.lp:.4f}',
            'sp': f'{c.sp:.4f}',
            'ws': c.whipsaws,
            'ws_color': 'pos' if c.whipsaws == 0 else ('warn' if c.whipsaws <= 5 else 'neg'),
            'dur': f'{c.duration_candles} candles',
            'worst': (f'-${abs(c.peak_intra_loss):,.2f}' if c.peak_intra_loss < 0 else f'${c.peak_intra_loss:,.2f}'),
            'worst_neg': c.peak_intra_loss < 0,
            'exit': exit_str,
            'exit_long': 'LONG' in exit_str,
            'liq': liq,
            'aborted': aborted,
            'start': s_dt,
            'end': e_dt,
        })
    return rows


def _build_ws_breakdown(result) -> list:
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
        color = 'pos' if n == 0 else ('warm' if n <= 5 else 'neg')
        rows.append({
            'n': n,
            'cnt': cnt,
            'pct': f'{pct:.1f}%',
            'cum': f'{cum_pct:.1f}%',
            'bar_w': round(bar_w, 1),
            'color': color,
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

        mkt = d.get('market_type', 'USDM').lower()

        upd(5, f'Fetching {trade_interval} candles…')
        trading_klines, err1 = fetch_historical_ohlcv(
            d['symbol'], trade_interval, days=bt_days,
            progress_callback=lambda p: upd(int(5 + p * 35), f'Fetching {trade_interval} candles…'),
            market_type=mkt,
        )
        if err1:
            raise RuntimeError(f'Trading candles: {err1}')

        upd(40, f'Fetching {atr_interval} ATR candles…')
        atr_klines, err2 = fetch_historical_ohlcv(
            d['symbol'], atr_interval, days=bt_days,
            progress_callback=lambda p: upd(int(40 + p * 35), f'Fetching {atr_interval} candles…'),
            market_type=mkt,
        )
        if err2:
            raise RuntimeError(f'ATR candles: {err2}')

        trading_candles = klines_to_candles(trading_klines)
        atr_candles = klines_to_candles(atr_klines)

        upd(78, 'Running CHV backtest…')
        result = run_backtest(
            symbol=d['symbol'],
            trading_candles=trading_candles,
            atr_candles=atr_candles,
            atr_period=int(d['atr_period']),
            base_lots=d['base_lots'],
            leverage=int(d['leverage']),
            capital=d['capital'],
            fee_rate=d['fee_rate_pct'] / 100.0,
            buffer=d['efficiency_buffer'],
            slippage_pct=d.get('slip_pct', 0.0005),
            reward_ratio=d['reward_ratio'],
            max_whipsaws=d['ws_limit'] if d['ws_limit_on'] else 0,
            atr_guard=d.get('atr_guard_on', True),
            atr_guard_multiplier=d.get('atr_guard_multiplier', 1.0),
        )
        upd(92, 'Saving CSV…')
        _save_bt_csv(result, d)

        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status='done', progress=100, msg='Done!',
                result=result, error=None, bt_capital=d['capital'],
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
        coinm_symbols=get_coinm_symbols(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
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
    bt_cap = job.get('bt_capital', d_job.get('capital', 1000.0))

    d = _parse_sidebar({}, session)
    d.update({k: v for k, v in d_job.items()
              if k not in ('slip_pct', 'base_asset', 'settle_coin', 'lot_min', 'lot_step', 'lot_dec')})

    total_trades = sum(len(c.steps) for c in result.cycles)
    growth_pct = (result.total_net_pnl / bt_cap * 100) if bt_cap else 0
    top6_ws = sorted([c.whipsaws for c in result.cycles], reverse=True)[:6]
    ws_breakdown = _build_ws_breakdown(result)

    # capital at worst_intra_loss_cycle
    cap_running = bt_cap
    cap_at_worst = bt_cap
    for c in result.cycles:
        if c.cycle_num == result.worst_intra_loss_cycle:
            cap_at_worst = cap_running
        cap_running += c.net_pnl

    log = _prep_bt_log(result, bt_cap, d_job.get('trading_tf', '1h'))
    charts = _build_bt_charts(result, bt_cap)

    return render_template('simulator/backtest.html',
                           bt_job_id=job_id,
                           bt_error=None,
                           result=result,
                           bt_log=log,
                           bt_charts=charts,
                           bt_cap=bt_cap,
                           total_trades=total_trades,
                           growth_pct=growth_pct,
                           top6_ws=top6_ws,
                           ws_breakdown=ws_breakdown,
                           cap_at_worst=cap_at_worst,
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
    return jsonify(get_available_symbols())


@app.route('/sim/api/price-atr')
def api_price_atr():
    sym = request.args.get('symbol', 'SOLUSDT')
    atr_tf = request.args.get('atr_tf', '4h')
    period = int(request.args.get('atr_period', 5))
    market_type = request.args.get('market_type', 'usdm').lower()
    try:
        price, atr, err = fetch_price_and_atr(sym, atr_tf, period, market_type)
        if err:
            return jsonify({'ok': False, 'error': err})
        return jsonify({'ok': True, 'price': price, 'atr': atr})
    except Exception as ex:
        return jsonify({'ok': False, 'error': str(ex)})


@app.route('/sim/sw.js')
def service_worker():
    return Response('// CHV Sim SW', mimetype='application/javascript')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8503, debug=False, threaded=True)
