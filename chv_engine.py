"""CHV Recovery Trade — core calculation engine shared by simulator and skill scripts."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import itertools


@dataclass
class CHVParams:
    symbol: str
    current_price: float
    atr_macro: float
    efficiency_buffer: float
    c: float
    d: float
    lp: float
    sp: float
    tl: float
    ts: float
    footprint: float
    footprint_pct: float
    is_safe: bool
    safety_status: str  # SAFE | CAUTION | ATR_TRAP
    safety_message: str


@dataclass
class LedgerStep:
    step: int
    trigger_price: float
    direction: str  # LONG | SHORT | EXIT
    execution: str
    active_lots: float
    closed_pnl: float
    running_balance: float
    margin_required: float
    is_exit: bool = False


@dataclass
class SimResult:
    params: CHVParams
    steps: List[LedgerStep]
    resolved: bool
    resolution_step: Optional[int]
    net_pnl: float
    steps_covered: int
    danger_step: Optional[int]
    danger_margin: Optional[float]
    capital_survives: bool
    peak_margin: float
    peak_margin_step: int


def calculate_params(
    symbol: str,
    price: float,
    atr: float,
    buffer: float = 0.8,
    ratio: float = 2.5,
    atr_guard_multiplier: float = 1.0,
) -> CHVParams:
    c = (atr * buffer) / 3.0
    d = ratio * c
    footprint = c + d
    footprint_pct = footprint / atr if atr > 0 else 999

    lp = price
    sp = price - c
    tl = lp + d
    ts = sp - d

    trap_threshold = atr * atr_guard_multiplier
    caution_threshold = trap_threshold * 0.95

    if footprint > trap_threshold:
        status = "ATR_TRAP"
        msg = f"ATR TRAP: footprint {footprint:.4f} exceeds ATR × {atr_guard_multiplier:.1f} ({trap_threshold:.4f}). Do not execute."
    elif footprint > caution_threshold:
        status = "CAUTION"
        msg = f"Footprint is {footprint_pct*100:.1f}% of ATR — tight. Consider 0.8 buffer or lower leverage."
    else:
        status = "SAFE"
        msg = f"Footprint {footprint:.4f} = {footprint_pct*100:.1f}% of ATR. Structure verified."

    return CHVParams(
        symbol=symbol,
        current_price=price,
        atr_macro=atr,
        efficiency_buffer=buffer,
        c=round(c, 6),
        d=round(d, 6),
        lp=round(lp, 6),
        sp=round(sp, 6),
        tl=round(tl, 6),
        ts=round(ts, 6),
        footprint=round(footprint, 6),
        footprint_pct=round(footprint_pct, 4),
        is_safe=(status != "ATR_TRAP"),
        safety_status=status,
        safety_message=msg,
    )


def simulate(
    params: CHVParams,
    base_lots: float,
    leverage: float,
    capital: float,
    max_steps: int = 15,
    fee_rate: float = 0.001,
) -> SimResult:
    steps: List[LedgerStep] = []
    balance = 0.0
    lots = base_lots
    direction = "LONG"
    resolved = False
    resolution_step = None

    margin_1 = (lots * params.lp) / leverage
    steps.append(LedgerStep(
        step=1, trigger_price=params.lp, direction="LONG",
        execution="Open Primary LONG", active_lots=round(lots, 6),
        closed_pnl=0.0, running_balance=0.0,
        margin_required=round(margin_1, 4),
    ))

    for n in range(2, max_steps + 2):
        if direction == "LONG":
            trigger = params.sp
            pnl = -(lots * params.c * leverage) - (lots * trigger * fee_rate)
            balance += pnl
            # Correct lot formula: first inversion (n==2) = base_lots × 0.5
            # Subsequent inversions = previous_lots × 1.5
            next_lots = round(base_lots * 0.5, 6) if n == 2 else round(lots * 1.5, 6)
            steps.append(LedgerStep(
                step=n, trigger_price=trigger, direction="SHORT",
                execution="Invert to SHORT (SL Hit)",
                active_lots=next_lots, closed_pnl=round(pnl, 4),
                running_balance=round(balance, 4),
                margin_required=round((next_lots * trigger) / leverage, 4),
            ))
            lots = next_lots
            direction = "SHORT"
        else:
            trigger = params.lp
            pnl = -(lots * params.c * leverage) - (lots * trigger * fee_rate)
            balance += pnl
            # Correct lot formula: first inversion (n==2) = base_lots × 0.5
            # Subsequent inversions = previous_lots × 1.5
            next_lots = round(base_lots * 0.5, 6) if n == 2 else round(lots * 1.5, 6)
            steps.append(LedgerStep(
                step=n, trigger_price=trigger, direction="LONG",
                execution="Invert to LONG (SL Hit)",
                active_lots=next_lots, closed_pnl=round(pnl, 4),
                running_balance=round(balance, 4),
                margin_required=round((next_lots * trigger) / leverage, 4),
            ))
            lots = next_lots
            direction = "LONG"

        # Test if current leg TP produces net positive
        ep = params.tl if direction == "LONG" else params.ts
        exit_pnl = (lots * params.d * leverage) - (lots * ep * fee_rate)
        if balance + exit_pnl > 0:
            final = round(balance + exit_pnl, 4)
            steps.append(LedgerStep(
                step=n + 1, trigger_price=ep,
                direction="EXIT",
                execution=f"{'TL' if direction == 'LONG' else 'TS'} Hit — CLOSE ALL",
                active_lots=0.0, closed_pnl=round(exit_pnl, 4),
                running_balance=final,
                margin_required=0.0, is_exit=True,
            ))
            balance = final
            resolved = True
            resolution_step = n + 1
            break

    # Capital analysis
    active_steps = [s for s in steps if not s.is_exit]
    margins = [(s.step, s.margin_required) for s in active_steps]
    peak_step, peak_margin = max(margins, key=lambda x: x[1]) if margins else (0, 0)

    danger_step = None
    danger_margin = None
    capital_survives = True

    for s, m in margins:
        if m > capital:
            danger_step = s
            danger_margin = m
            capital_survives = False
            break

    # Theoretical capacity: how many whipsaws capital can fund before hitting
    # either the margin limit (can't open next position) or the accumulated-loss
    # limit (total realized losses exceed capital) — whichever comes first.
    # Uses correct lot formula: first inversion = base_lots × 0.5, then × 1.5.
    theo_lots = base_lots
    steps_covered = 0
    accumulated_loss = 0.0
    entry_price = max(params.lp, params.sp)
    is_first_inversion = True
    for _ in range(300):
        accumulated_loss += theo_lots * params.c * leverage
        if is_first_inversion:
            theo_lots = round(base_lots * 0.5, 6)
            is_first_inversion = False
        else:
            theo_lots = round(theo_lots * 1.5, 6)
        margin = (theo_lots * entry_price) / leverage
        if margin > capital or accumulated_loss > capital:
            break
        steps_covered += 1

    return SimResult(
        params=params,
        steps=steps,
        resolved=resolved,
        resolution_step=resolution_step,
        net_pnl=round(balance, 4),
        steps_covered=steps_covered,
        danger_step=danger_step,
        danger_margin=danger_margin,
        capital_survives=capital_survives,
        peak_margin=round(peak_margin, 4),
        peak_margin_step=peak_step,
    )


@dataclass
class OptimResult:
    buffer: float
    leverage: float
    base_lots: float
    net_pnl: float
    steps_covered: int
    capital_survives: bool
    footprint_pct: float
    safety_status: str
    score: float


def optimize(
    symbol: str,
    price: float,
    atr: float,
    capital: float,
    fee_rate: float = 0.001,
    buffers: List[float] = None,
    leverages: List[float] = None,
    lot_sizes: List[float] = None,
    max_steps: int = 15,
) -> List[OptimResult]:
    if buffers is None:
        buffers = [0.7, 0.75, 0.8, 0.85, 0.9]
    if leverages is None:
        leverages = [5, 10, 20, 25, 50]
    if lot_sizes is None:
        lot_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]

    results = []
    for buf, lev, lots in itertools.product(buffers, leverages, lot_sizes):
        params = calculate_params(symbol, price, atr, buf)
        if not params.is_safe:
            continue
        sim = simulate(params, lots, lev, capital, max_steps, fee_rate)
        # Score: positive pnl + more steps + capital safety weighted
        score = (
            (sim.net_pnl if sim.net_pnl > 0 else -9999)
            + sim.steps_covered * 10
            + (500 if sim.capital_survives else 0)
            - (params.footprint_pct * 100)
        )
        results.append(OptimResult(
            buffer=buf,
            leverage=lev,
            base_lots=lots,
            net_pnl=sim.net_pnl,
            steps_covered=sim.steps_covered,
            capital_survives=sim.capital_survives,
            footprint_pct=params.footprint_pct,
            safety_status=params.safety_status,
            score=round(score, 2),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def generate_bot_config(
    params: CHVParams,
    base_lots: float,
    leverage: float,
    capital: float,
    fee_rate: float = 0.001,
    exchange: str = "binance",
) -> dict:
    from datetime import datetime
    return {
        "_meta": {
            "system": "CHV Recovery Trade",
            "architect": "Chitti Vijakkhana",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
        },
        "exchange": {
            "name": exchange,
            "type": "futures",
            "api_key": "[FILL IN]",
            "api_secret": "[FILL IN]",
            "testnet": True,
        },
        "strategy": {
            "name": "CHV_RECOVERY",
            "symbol": params.symbol,
            "market_type": "FUTURES_USDT",
            "active": False,
        },
        "parameters": {
            "cut_loss_zone_c": params.c,
            "recovery_target_d": params.d,
            "efficiency_buffer": params.efficiency_buffer,
            "atr_macro_reference": params.atr_macro,
            "lot_multiplier": 1.5,
            "footprint": params.footprint,
            "footprint_pct_atr": params.footprint_pct,
        },
        "entry_levels": {
            "LP_long_entry": params.lp,
            "SP_short_entry": params.sp,
            "TL_take_profit_long": params.tl,
            "TS_take_profit_short": params.ts,
        },
        "position_sizing": {
            "base_lots": base_lots,
            "lot_multiplier": 1.5,
            "leverage": leverage,
            "fee_rate": fee_rate,
        },
        "risk_management": {
            "starting_capital_usd": capital,
            "abort_if_balance_below_usd": round(capital * 0.1, 2),
            "max_active_whipsaw_steps": 10,
            "atr_trap_guard": True,
        },
        "execution": {
            "order_type": "LIMIT",
            "time_in_force": "GTC",
            "hedge_mode": True,
            "reduce_only_on_exit": True,
        },
        "expansion": {
            "supported_exchanges": ["binance", "bybit", "okx"],
            "quantconnect_bridge": False,
        },
    }
