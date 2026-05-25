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
                          get_coinm_symbols, coinm_contract_size, coinm_base_asset)

app = Flask(__name__, template_folder=str(SIM_DIR / 'templates'),
            static_folder=str(SIM_DIR / 'static'))
app.secret_key = 'chv-sim-flask-2025-x7k'

BACKTEST_DIR = SIM_DIR / 'backtests'
BACKTEST_DIR.mkdir(exist_ok=True)

_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()

# ── Constants ─────────────────────────────────────────────────────────────────
SLIP_LABELS   = ['None (0%)', '0.01%', '0.02%', '0.05%', '0.10%', '0.20%']
SLIP_VALUES   = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
BUFFER_CHOICES= [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TF_CHOICES    = ['1m','5m','15m','30m','1h','4h','12h','1d']
LOT_SPECS     = {
    'BTCUSDT':10,'ETHUSDT':1,'SOLUSDT':1,'BNBUSDT':1,'XRPUSDT':1,
    'FARTCOINUSDT':1,'AVAXUSDT':1,'SUIUSDT':1,'DOGEUSDT':1,
    'NEARUSDT':1,'JUPUSDT':1,'JTOUSDT':1,'ONDOUSDT':1,'XAUUSDT':1,
    # CoinM — all in contracts
    'BTCUSD_PERP':1,'ETHUSD_PERP':1,'BNBUSD_PERP':1,'XRPUSD_PERP':1,
    'ADAUSD_PERP':1,'SOLUSD_PERP':1,'DOTUSD_PERP':1,'LINKUSD_PERP':1,
    'LTCUSD_PERP':1,'BCHUSD_PERP':1,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _sidebar_defaults():
    return dict(
        market_type='usdm', symbol='SOLUSDT', trading_tf='1h', atr_tf='1h',
        atr_period=14, price_val=0.0, atr_val=0.0,
        efficiency_buffer=0.7, reward_ratio=2.5,
        base_lots=10.0, leverage=20, capital=1000.0,
        fee_rate_pct=0.05, slippage_label='None (0%)',
        ws_limit_on=False, ws_limit=10,
        atr_guard_on=False, atr_guard_multiplier=2.0,
        use_live=True, base_asset='SOL', settle_coin='USDT',
        lot_dec=1, notional_ok='', notional_warn='',
        calc_ws=10, atr_mult=1.0,
    )

def _parse_sidebar(form, sess):
    d = _sidebar_defaults()
    src = {**sess, **form}
    mt = src.get('market_type', 'usdm')
    d['market_type'] = mt

    sym = src.get('symbol', d['symbol'])
    d['symbol'] = sym

    if mt == 'coinm':
        d['base_asset']  = coinm_base_asset(sym)
        d['settle_coin'] = d['base_asset']
    else:
        d['base_asset']  = sym.replace('USDT','').replace('BUSD','')
        d['settle_coin'] = 'USDT'

    for key in ('trading_tf','atr_tf'):
        if src.get(key): d[key] = src[key]
    for key in ('atr_period','leverage','ws_limit'):
        try: d[key] = int(src[key])
        except: pass
    for key in ('price_val','atr_val','efficiency_buffer','reward_ratio',
                'base_lots','capital','fee_rate_pct','atr_guard_multiplier',
                'calc_ws','atr_mult'):
        try: d[key] = float(src[key])
        except: pass
    for key in ('ws_limit_on','atr_guard_on','use_live'):
        d[key] = bool(src.get(key) and src[key] not in ('','off','false','0'))

    d['slippage_label'] = src.get('slippage_label', d['slippage_label'])
    try:
        d['slip_pct'] = SLIP_VALUES[SLIP_LABELS.index(d['slippage_label'])]
    except: d['slip_pct'] = 0.0

    # Lot decimals
    lot_min = LOT_SPECS.get(sym, 1)
    d['lot_dec'] = 0 if lot_min >= 1 else (1 if lot_min >= 0.1 else 2)

    # Validate notional for CoinM
    if mt == 'coinm':
        cs = coinm_contract_size(sym)
        _dec = 6 if d['settle_coin'] == 'BTC' else (4 if d['settle_coin'] in ('ETH','BNB') else 2)
        if d['price_val'] > 0:
            first_inv = d['base_lots'] * 0.5 * cs / d['price_val']
            d['notional_ok'] = f"First inv: {d['base_lots'] * 0.5:.0f} contracts = {first_inv:.{_dec}f} {d['settle_coin']}"
    return d

def _save_to_session(d):
    skip = {'notional_ok','notional_warn','lot_dec','slip_pct','base_asset','settle_coin'}
    for k, v in d.items():
        if k not in skip:
            session[k] = v

# ── Chart helpers ─────────────────────────────────────────────────────────────
def _svg_path(xs, ys, x0, x1, y0, y1, y_min, y_max):
    if not xs: return '', ''
    y_rng = y_max - y_min or 1
    def px(i): return round(x0 + (i / (len(xs)-1 or 1)) * (x1-x0), 1)
    def py(v): return round(y1 + ((v - y_min) / y_rng) * (y0 - y1), 1)
    pts = [(px(i), py(v)) for i, v in zip(xs, ys)]
    line = 'M ' + ' L '.join(f'{x},{y}' for x,y in pts)
    area = line + f' L {pts[-1][0]},{y0} L {pts[0][0]},{y0} Z'
    return line, area

def _y_labels(y_min, y_max, y_bot, y_top, steps=4, coin='$'):
    y_range = y_max - y_min or 1
    if coin == '$':
        if abs(y_max) >= 1000:  fmt = lambda v: f"${v/1000:.1f}k"
        elif abs(y_max) >= 100: fmt = lambda v: f"${v:.0f}"
        elif abs(y_max) >= 10:  fmt = lambda v: f"${v:.1f}"
        else:                    fmt = lambda v: f"${v:.2f}"
    else:
        _cdec = 6 if coin == 'BTC' else (4 if coin in ('ETH','BNB') else 2)
        fmt = lambda v, d=_cdec: f"{v:,.{d}f}"
    out = []
    for i in range(steps + 1):
        val = y_min + y_range * i / steps
        yp  = y_bot + ((val - y_min) / y_range) * (y_top - y_bot)
        out.append({'y': round(yp,1), 'label': fmt(val)})
    return out

def _x_labels(n, x0, x1, steps=5):
    out = []
    for i in range(steps + 1):
        val = n * i / steps
        xp  = x0 + (i / steps) * (x1 - x0)
        out.append({'x': round(xp,1), 'label': str(int(val))})
    return out

def _build_bt_charts(result, capital: float, settle_coin: str = 'USDT') -> dict:
    coin = '$' if settle_coin in ('USDT','USD','') else settle_coin
    cycles = result.cycles
    if not cycles: return {}
    n = len(cycles)
    x0, x1 = 40, 710

    # Equity curve (capital after each cycle)
    eq_vals = [capital]
    for c in cycles: eq_vals.append(eq_vals[-1] + c.net_pnl)
    eq_min = min(eq_vals) * 0.98; eq_max = max(eq_vals) * 1.02
    eq_rng = eq_max - eq_min or 1
    def py_eq(v): return 240 + ((v-eq_min)/eq_rng)*(40-240)
    eq_pts = [(round(x0+(i/(n or 1))*(x1-x0),1), round(py_eq(v),1)) for i,v in enumerate(eq_vals)]
    eq_line = 'M '+' L '.join(f'{x},{y}' for x,y in eq_pts)
    eq_area = eq_line + f' L {eq_pts[-1][0]},240 L {eq_pts[0][0]},240 Z'
    eq_dot  = eq_pts[-1]

    # Capital per cycle (end-of-cycle only)
    cap_vals = list(eq_vals[1:])
    cap_min = min(cap_vals)*0.98; cap_max = max(cap_vals)*1.02
    cap_rng = cap_max - cap_min or 1
    def py_cap(v): return 240+((v-cap_min)/cap_rng)*(40-240)
    cap_pts = [(round(x0+(i/(n-1 or 1))*(x1-x0),1), round(py_cap(v),1)) for i,v in enumerate(cap_vals)]
    cap_line = 'M '+' L '.join(f'{x},{y}' for x,y in cap_pts)
    cap_area = cap_line + f' L {cap_pts[-1][0]},240 L {cap_pts[0][0]},240 Z'

    # Capital per trade (including all WS steps)
    ct_vals = [capital]
    for c in cycles:
        running = ct_vals[-1]
        for step in c.steps:
            running += step.pnl
            ct_vals.append(running)
    ct_min = min(ct_vals)*0.98; ct_max = max(ct_vals)*1.02
    ct_rng = ct_max - ct_min or 1
    m = len(ct_vals)
    def py_ct(v): return 240+((v-ct_min)/ct_rng)*(40-240)
    ct_pts = [(round(x0+(i/(m-1 or 1))*(x1-x0),1), round(py_ct(v),1)) for i,v in enumerate(ct_vals)]
    ct_line = 'M '+' L '.join(f'{x},{y}' for x,y in ct_pts)
    ct_area = ct_line + f' L {ct_pts[-1][0]},240 L {ct_pts[0][0]},240 Z'

    # Whipsaws per cycle bar chart
    ws_vals = [c.whipsaws for c in cycles]
    ws_max  = max(ws_vals) if ws_vals else 1
    bar_w   = max(1, (x1-x0) / n - 1)
    ws_bars = []
    for i, w in enumerate(ws_vals):
        bx = x0 + i*(x1-x0)/n
        bh = (w/ws_max)*180 if ws_max else 0
        col = 'pos' if w==0 else ('warn' if w<=5 else 'neg')
        ws_bars.append({'x':round(bx,1),'w':round(bar_w,1),'h':round(bh,1),'color':col,'ws':w})

    return dict(
        eq_line=eq_line, eq_area=eq_area, eq_dot=eq_dot,
        eq_y_lbls=_y_labels(eq_min,eq_max,240,40,4,coin=coin),
        eq_x_lbls=_x_labels(n,x0,x1),
        cap_line=cap_line, cap_area=cap_area,
        cap_y_lbls=_y_labels(cap_min,cap_max,240,40,4,coin=coin),
        cap_x_lbls=_x_labels(n,x0,x1),
        ct_line=ct_line, ct_area=ct_area,
        ct_y_lbls=_y_labels(ct_min,ct_max,240,40,4,coin=coin),
        ct_x_lbls=_x_labels(m,x0,x1),
        ws_bars=ws_bars, ws_max=ws_max,
    )

def _prep_bt_log(result, capital: float, trading_tf: str, settle_coin: str = 'USDT') -> list:
    TF_MS = {'1m':60000,'5m':300000,'15m':900000,'30m':1800000,
             '1h':3600000,'4h':14400000,'12h':43200000,'1d':86400000}
    tf_ms = TF_MS.get(trading_tf, 3600000)
    cap = capital
    coinm = settle_coin not in ('USDT','USD','')
    _dec = 6 if settle_coin=='BTC' else (4 if settle_coin in ('ETH','BNB') else 2)
    def fmt_coin(v):
        if coinm: return f"{v:,.{_dec}f} {settle_coin}"
        return f"+${abs(v):,.2f}" if v >= 0 else f"-${abs(v):,.2f}"
    def fmt_cap(v):
        if coinm: return f"{v:,.{_dec}f} {settle_coin}"
        return f"${v:,.2f}"
    rows = []
    for c in result.cycles:
        cap += c.net_pnl
        dur_ms = (c.end_ts or 0) - (c.start_ts or 0)
        dur_c  = max(1, round(dur_ms / tf_ms))
        s_dt = datetime.datetime.utcfromtimestamp(c.start_ts/1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
        e_dt = datetime.datetime.utcfromtimestamp(c.end_ts/1000).strftime('%Y-%m-%d %H:%M')   if c.end_ts   else '—'
        liq      = not c.capital_ok
        aborted  = c.exit_direction == 'ABORTED'
        exit_str = '💀 LIQ' if liq else ('⛔ ABORTED' if aborted else c.exit_direction)
        rows.append({
            'num':      c.cycle_num,
            'capital':  fmt_cap(cap),
            'pnl':      c.net_pnl,
            'pnl_fmt':  fmt_coin(c.net_pnl),
            'pnl_pos':  c.net_pnl >= 0,
            'entry':    f"{c.entry_price:.4f}",
            'atr':      f"{c.atr_at_entry:.4f}",
            'lp':       f"{c.lp:.4f}",
            'sp':       f"{c.sp:.4f}",
            'ws':       c.whipsaws,
            'ws_color': 'pos' if c.whipsaws==0 else ('warn' if c.whipsaws<=5 else 'neg'),
            'dur':      f"{dur_c} candles",
            'worst':    fmt_coin(c.peak_intra_loss),
            'worst_neg':c.peak_intra_loss < 0,
            'exit':     exit_str,
            'liq':      liq,
            'stopped':  stopped,
            'start':    s_dt,
            'end':      e_dt,
        })
    return list(reversed(rows))

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/sim')
@app.route('/sim/')
def sim_root(): return redirect(url_for('guide'))

@app.route('/sim/guide')
def guide():
    d = _parse_sidebar(request.form if request.method=='POST' else {}, session)
    _save_to_session(d)
    return render_template('simulator/guide.html', mode='guide', **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/about')
def about():
    d = _parse_sidebar({}, session)
    return render_template('simulator/about.html', mode='about', **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/simulator', methods=['GET','POST'])
def simulator():
    d = _parse_sidebar(request.form if request.method=='POST' else {}, session)
    _save_to_session(d)
    result = None; error = None
    if request.method == 'POST' and d['price_val'] > 0:
        try:
            p = calculate_params(d['symbol'], d['price_val'], d['atr_val'],
                                 d['efficiency_buffer'], d['reward_ratio'])
            result = simulate(p,
                base_lots=d['base_lots'], leverage=d['leverage'],
                capital=d['capital'], fee_rate=d['fee_rate_pct']/100,
            )
        except Exception as ex: error = str(ex)
    return render_template('simulator/simulator.html', mode='simulator',
                           result=result, error=error, **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/backtest', methods=['GET','POST'])
def backtest():
    d = _parse_sidebar(request.form if request.method=='POST' else {}, session)
    _save_to_session(d)
    ctx = dict(**d,
               mode='backtest', bt_job_id=None, bt_error=None,
               result=None, bt_log=[], bt_charts={},
               bt_days=90, total_trades=0, fee_pct_used=d['fee_rate_pct'],
               top6_ws=[], wil_ns_cap='—',
               slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
               tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)
    if request.method == 'POST' and request.form.get('action') == 'run':
        try:
            bt_days = int(request.form.get('bt_days', 90))
            ctx['bt_days'] = bt_days
            d['bt_days']   = bt_days
            bt_cap = d['capital']
            job_id = str(uuid.uuid4())[:8]
            with _JOBS_LOCK:
                _JOBS[job_id] = {'status':'queued','progress':0,'msg':'Queued…',
                                 'result':None,'error':None,'d':dict(d,bt_days=bt_days)}
            t = threading.Thread(target=_bt_worker, args=(job_id,), daemon=True)
            t.start()
            ctx['bt_job_id'] = job_id
        except Exception as ex:
            ctx['bt_error'] = str(ex)
    return render_template('simulator/backtest.html', **ctx)

def _bt_worker(job_id: str):
    with _JOBS_LOCK:
        d = dict(_JOBS[job_id]['d'])
    def upd(p, msg):
        with _JOBS_LOCK: _JOBS[job_id].update(progress=p, msg=msg, status='running')
    try:
        upd(5, 'Fetching candles…')
        mt = d.get('market_type','usdm')
        klines, kl_err = fetch_historical_ohlcv(
            d['symbol'], d['atr_tf'], days=d['bt_days'],
            market_type=mt)
        if kl_err: raise RuntimeError(kl_err)
        atr_candles = klines_to_candles(klines)
        if d['trading_tf'] != d['atr_tf']:
            klines2, kl_err2 = fetch_historical_ohlcv(d['symbol'], d['trading_tf'],
                                             days=d['bt_days'], market_type=mt)
            if kl_err2: raise RuntimeError(kl_err2)
            trading_candles = klines_to_candles(klines2)
        else:
            trading_candles = atr_candles
        upd(30, 'Running backtest…')
        bt_capital = d['capital']
        result = run_backtest(
            symbol=d['symbol'], trading_candles=trading_candles,
            atr_candles=atr_candles, atr_period=int(d['atr_period']),
            buffer=d['efficiency_buffer'],
            reward_ratio=d['reward_ratio'],
            base_lots=d['base_lots'], leverage=int(d['leverage']),
            capital=bt_capital, fee_rate=d['fee_rate_pct']/100,
            slippage_pct=d.get('slip_pct',0)/100,
            max_whipsaws=d['ws_limit'] if d['ws_limit_on'] else 0,
            atr_guard=d.get('atr_guard_on',False),
            atr_guard_multiplier=d.get('atr_guard_multiplier',2.0),
        )
        upd(90, 'Saving…')
        _save_bt_csv(result, d)
        with _JOBS_LOCK:
            _JOBS[job_id].update(status='done', progress=100, msg='Done!',
                                 result=result, error=None, bt_capital=bt_capital)
    except Exception as ex:
        with _JOBS_LOCK:
            _JOBS[job_id].update(status='error', progress=0, msg=str(ex),
                                 result=None, error=str(ex))

def _save_bt_csv(result, d: dict):
    meta = {'symbol':d['symbol'],'trading_tf':d['trading_tf'],
            'settle_coin':d.get('settle_coin','USDT'),
            'leverage':d['leverage'],'base_lots':d['base_lots'],
            'capital':d['capital'],'total_cycles':result.total_cycles,
            'winning_cycles':result.winning_cycles,
            'total_net_pnl':result.total_net_pnl,
            'liquidated':result.liquidated,
            'saved_utc':datetime.datetime.utcnow().isoformat()}
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    fpath = BACKTEST_DIR / f"chv_backtest_{d['symbol']}_{ts}.csv"
    settle = d.get('settle_coin','USDT')
    coinm = settle not in ('USDT','USD','')
    _dec = 6 if settle=='BTC' else (4 if settle in ('ETH','BNB') else 2)
    def fc(v):
        if coinm: return f"{v:,.{_dec}f} {settle}"
        return f"+${v:,.2f}" if v>=0 else f"-${abs(v):,.2f}"
    cap = d['capital']
    with open(fpath,'w',newline='') as f:
        f.write('# '+json.dumps(meta)+'\n')
        w = csv.writer(f)
        w.writerow(['#','Capital After','P&L','Entry','ATR','LP','SP',
                    'Whipsaws','Duration','Worst Loss','Exit','Start','End'])
        for c in result.cycles:
            cap += c.net_pnl
            sdt = datetime.datetime.utcfromtimestamp(c.start_ts/1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
            edt = datetime.datetime.utcfromtimestamp(c.end_ts/1000).strftime('%Y-%m-%d %H:%M')   if c.end_ts   else '—'
            exit_str = 'LIQ' if not c.capital_ok else getattr(c,'exit_direction','—')
            w.writerow([c.cycle_num,fc(cap),fc(c.net_pnl),
                        f"{c.entry_price:.4f}",f"{c.atr_at_entry:.4f}",
                        f"{c.lp:.4f}",f"{c.sp:.4f}",
                        c.whipsaws,f"{c.duration_candles} candles",
                        fc(c.peak_intra_loss),exit_str,sdt,edt])

@app.route('/sim/backtest/status/<job_id>')
def bt_status(job_id):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id,{}))
    return jsonify({'status':job.get('status','unknown'),
                    'progress':job.get('progress',0),
                    'msg':job.get('msg','')})

@app.route('/sim/backtest/result/<job_id>')
def bt_result(job_id):
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id,{}))
    if job.get('status') != 'done':
        return redirect(url_for('backtest'))
    result = job['result']
    d_job  = job.get('d',{})
    bt_cap = job.get('bt_capital', d_job.get('capital',0))
    d = _parse_sidebar({}, session)
    d.update({k:v for k,v in d_job.items()})
    sc = d_job.get('settle_coin','USDT')
    coinm = sc not in ('USDT','USD','')
    _dec = 6 if sc=='BTC' else (4 if sc in ('ETH','BNB') else 2)
    total_trades = sum(c.whipsaws+1 for c in result.cycles)
    top6_ws = sorted([c.whipsaws for c in result.cycles], reverse=True)[:6]
    log = _prep_bt_log(result, bt_cap, d_job.get('trading_tf','1h'), sc)
    # find capital at worst_intra_loss_cycle for sub-foot
    wil_cap = '—'
    for r in log:
        if r['num'] == result.worst_intra_loss_cycle:
            wil_cap = r['capital']; break
    return render_template('simulator/backtest.html',
               mode='backtest', result=result, bt_job_id=job_id,
               capital=bt_cap, settle_coin=sc, coinm=coinm,
               market_type=d_job.get('market_type','usdm'),
               base_asset=d_job.get('base_asset','SOL'),
               symbol=d_job.get('symbol','SOLUSDT'),
               trading_tf=d_job.get('trading_tf','1h'),
               bt_days=d_job.get('bt_days',90),
               bt_charts=_build_bt_charts(result, bt_cap, sc),
               bt_log=log, bt_error=None,
               fee_rate_pct=d_job.get('fee_rate_pct',0.05),
               total_trades=total_trades,
               fee_pct_used=d_job.get('fee_rate_pct',0.05),
               top6_ws=top6_ws, wil_ns_cap=wil_cap,
               slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
               tf_choices=TF_CHOICES, lot_specs=LOT_SPECS, **d)

@app.route('/sim/backtest/download-csv')
def bt_download_csv():
    job_id = request.args.get('job_id','')
    with _JOBS_LOCK:
        job = dict(_JOBS.get(job_id,{}))
    if not job or job.get('status') != 'done':
        return "Result not found",404
    result = job['result']
    d      = job.get('d',{})
    bt_cap = job.get('bt_capital', d.get('capital',0))
    settle = d.get('settle_coin','USDT')
    coinm  = settle not in ('USDT','USD','')
    _dec   = 6 if settle=='BTC' else (4 if settle in ('ETH','BNB') else 2)
    def fc(v):
        if coinm: return f"{v:,.{_dec}f} {settle}"
        return f"+${v:,.2f}" if v>=0 else f"-${abs(v):,.2f}"
    buf = io.StringIO()
    buf.write('# '+json.dumps({'symbol':d.get('symbol'),'settle_coin':settle,
        'exported_utc':datetime.datetime.utcnow().isoformat()})+'\n')
    w = csv.writer(buf)
    w.writerow(['#','Capital After','P&L','Entry','ATR','LP','SP',
                'Whipsaws','Duration','Worst Loss','Exit','Start','End'])
    cap = bt_cap
    for c in result.cycles:
        cap += c.net_pnl
        sdt = datetime.datetime.utcfromtimestamp(c.start_ts/1000).strftime('%Y-%m-%d %H:%M') if c.start_ts else '—'
        edt = datetime.datetime.utcfromtimestamp(c.end_ts/1000).strftime('%Y-%m-%d %H:%M')   if c.end_ts   else '—'
        w.writerow([c.cycle_num,fc(cap),fc(c.net_pnl),
                    f"{c.entry_price:.4f}",f"{c.atr_at_entry:.4f}",
                    f"{c.lp:.4f}",f"{c.sp:.4f}",
                    c.whipsaws,f"{c.duration_candles} candles",
                    fc(c.peak_intra_loss),getattr(c,'exit_direction','—'),sdt,edt])
    ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    name = f"chv_backtest_{d.get('symbol','unknown')}_{ts}.csv"
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':f'attachment; filename="{name}"'})

@app.route('/sim/optimizer', methods=['GET','POST'])
def optimizer_view():
    d = _parse_sidebar(request.form if request.method=='POST' else {}, session)
    _save_to_session(d)
    return render_template('simulator/optimizer.html', mode='optimizer', **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/botconfig', methods=['GET','POST'])
def botconfig():
    d = _parse_sidebar(request.form if request.method=='POST' else {}, session)
    _save_to_session(d)
    return render_template('simulator/botconfig.html', mode='botconfig', **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/history')
def history():
    files = sorted(BACKTEST_DIR.glob('chv_backtest_*.csv'), reverse=True)
    entries = []
    for f in files:
        try:
            with open(f) as fp: meta_line = fp.readline()
            meta = json.loads(meta_line.lstrip('#').strip())
        except: meta = {}
        entries.append({'name':f.name,'size':f.stat().st_size,**meta})
    d = _parse_sidebar({}, session)
    return render_template('simulator/history.html', mode='history',
                           entries=entries, **d,
                           slip_labels=SLIP_LABELS, buffer_choices=BUFFER_CHOICES,
                           tf_choices=TF_CHOICES, lot_specs=LOT_SPECS)

@app.route('/sim/history/download/<path:filename>')
def history_download(filename):
    fpath = BACKTEST_DIR / filename
    if not fpath.exists(): return "Not found",404
    return send_file(str(fpath), as_attachment=True, download_name=filename)

@app.route('/sim/history/delete', methods=['POST'])
def history_delete():
    fn = request.form.get('filename','')
    fp = BACKTEST_DIR / fn
    if fp.exists() and fp.suffix == '.csv': fp.unlink()
    return redirect(url_for('history'))

@app.route('/sim/history/clear', methods=['POST'])
def history_clear():
    for f in BACKTEST_DIR.glob('chv_backtest_*.csv'): f.unlink()
    return redirect(url_for('history'))

@app.route('/sim/api/symbols')
def api_symbols():
    mt = request.args.get('market_type','usdm')
    if mt == 'coinm': return jsonify(get_coinm_symbols())
    return jsonify(get_available_symbols())

@app.route('/sim/api/price-atr')
def api_price_atr():
    sym    = request.args.get('symbol','SOLUSDT')
    atr_tf = request.args.get('atr_tf','1h')
    period = int(request.args.get('atr_period',14))
    mt     = request.args.get('market_type','usdm')
    try:
        price, atr = fetch_price_and_atr(sym, atr_tf, period, market_type=mt)
        return jsonify({'price':price,'atr':atr,'ok':True})
    except Exception as ex:
        return jsonify({'ok':False,'error':str(ex)})

@app.route('/sim/api/lot-calc')
def api_lot_calc():
    sym     = request.args.get('symbol','SOLUSDT')
    price   = float(request.args.get('price',0) or 0)
    atr     = float(request.args.get('atr',0) or 0)
    buf     = float(request.args.get('buffer',0.7))
    rr      = float(request.args.get('reward_ratio',2.5))
    lev     = int(request.args.get('leverage',20))
    capital = float(request.args.get('capital',1000))
    calc_ws = int(request.args.get('calc_ws',10))
    atr_m   = float(request.args.get('atr_mult',1.0))
    mt      = request.args.get('market_type','usdm')
    if price <= 0 or atr <= 0:
        return jsonify({'ok':False,'error':'price/atr missing'})
    try:
        p = calculate_params(sym, price, atr, buf, rr)
        stressed_atr = atr * atr_m
        isCoinm = mt == 'coinm'
        if isCoinm:
            cs = coinm_contract_size(sym)
            base = coinm_base_asset(sym)
            cap_coin = capital
            def loss_per_ws(lots, ep, stop):
                return lots * cs * abs(1/stop - 1/ep)
            # find max safe lots
            max_lots = 1
            for lots in range(1, 500):
                running_loss = 0
                cur_lots = lots; cur_ep = price
                for ws in range(calc_ws + 1):
                    stop = cur_ep - stressed_atr if True else cur_ep + stressed_atr
                    running_loss += loss_per_ws(cur_lots, cur_ep, stop)
                    if running_loss >= cap_coin * 0.8: break
                    next_l = lots * 0.5 if ws == 0 else cur_lots * 1.5
                    cur_lots = next_l; cur_ep = stop
                else:
                    max_lots = lots; continue
                break
            return jsonify({'ok':True,'max_lots':max_lots,
                            'unit':'contracts','settle':base,
                            'c':round(p.c,6),'d':round(p.d,6),
                            'lp':round(p.lp,4),'sp':round(p.sp,4),
                            'tl':round(p.tl,4),'ts':round(p.ts,4)})
        else:
            lot_min = LOT_SPECS.get(sym,1)
            def loss_per_ws_usdm(lots, ep, stop):
                return lots * abs(stop - ep) * lev / ep * ep  # notional
            notional_per_lot = price * lot_min
            max_loss = capital * lev * 0.8
            # simple: find max lots where stress scenario stays solvent
            max_lots = lot_min
            for multiplier in range(1, 1000):
                lots = lot_min * multiplier
                running = 0
                cur = lots; cur_ep = price
                for ws in range(calc_ws + 1):
                    running += cur * stressed_atr
                    if running >= max_loss: break
                    cur = lots * 0.5 if ws==0 else cur * 1.5
                else:
                    max_lots = lots; continue
                break
            return jsonify({'ok':True,'max_lots':max_lots,'unit':'lots',
                            'c':round(p.c,6),'d':round(p.d,6),
                            'lp':round(p.lp,4),'sp':round(p.sp,4),
                            'tl':round(p.tl,4),'ts':round(p.ts,4)})
    except Exception as ex:
        return jsonify({'ok':False,'error':str(ex)})

@app.route('/sim/sw.js')
def service_worker():
    return Response("// CHV Sim SW", mimetype='application/javascript')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8503, debug=False, threaded=True)
