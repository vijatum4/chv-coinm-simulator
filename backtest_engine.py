"""CHV Recovery Trade — historical backtest engine."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import statistics


@dataclass
class Candle:
    timestamp: int   # ms
    open: float
    high: float
    low: float
    close: float


@dataclass
class CycleStep:
    direction: str   # LONG | SHORT | EXIT
    trigger_price: float
    lots: float
    pnl: float
    candle_idx: int
    timestamp: int


@dataclass
class Cycle:
    cycle_num: int
    entry_price: float
    atr_at_entry: float
    c: float
    d: float
    lp: float
    sp: float
    tl: float
    ts: float
    base_lots: float
    steps: List[CycleStep]
    whipsaws: int
    net_pnl: float
    start_idx: int
    end_idx: int
    start_ts: int
    end_ts: int
    duration_candles: int
    exit_direction: str   # LONG_TP | SHORT_TP | ABORTED
    capital_ok: bool
    peak_intra_loss: float  # worst accumulated loss before TP hit


@dataclass
class BacktestResult:
    symbol: str
    trading_tf: str
    atr_tf: str
    period_start: int
    period_end: int
    total_cycles: int
    winning_cycles: int
    aborted_cycles: int
    total_net_pnl: float
    total_fees: float
    avg_whipsaws: float
    max_whipsaws: int
    avg_duration_candles: float
    peak_equity: float
    max_drawdown: float           # worst cross-cycle equity dip
    max_drawdown_cycle: int
    worst_intra_loss: float       # worst single-cycle accumulated loss before TP
    worst_intra_loss_cycle: int
    liquidated: bool
    liquidation_cycle: int        # cycle number where account blew
    liquidation_step: int         # whipsaw step inside that cycle
    liquidation_loss: float       # accumulated loss at point of liquidation
    data_exhausted: bool          # True if backtest ran out of candles naturally
    equity_curve: List[float]
    cycle_pnls: List[float]
    cycles: List[Cycle]


def calc_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        trs.append(tr)
    return statistics.mean(trs[-period:])


def align_atr_candles(
    trading_candles: List[Candle],
    atr_candles: List[Candle],
    trading_idx: int,
) -> List[Candle]:
    """Return ATR candles up to and including the current trading candle timestamp."""
    ts = trading_candles[trading_idx].timestamp
    return [c for c in atr_candles if c.timestamp <= ts]


def simulate_cycle_on_candles(
    candles: List[Candle],
    start_idx: int,
    lp: float,
    sp: float,
    tl: float,
    ts_level: float,
    c: float,
    d: float,
    base_lots: float,
    leverage: float,
    capital: float,
    fee_rate: float,
    slippage_pct: float = 0.0,
    start_direction: str = "LONG",
    max_whipsaws: int = 0,
) -> tuple[List[CycleStep], int, float, bool, float, int, float]:
    """
    Simulate one CHV cycle starting at start_idx.
    start_direction: "LONG" opens at LP, "SHORT" opens at SP (momentum continuation).
    Returns (steps, end_idx, net_pnl, capital_ok, peak_intra_loss, liquidation_step, total_fees).
    liquidation_step = whipsaw number where capital ran out (0 if no liquidation).
    Slippage is modelled as an additional cost per fill: lots × price × slippage_pct.
    """
    steps: List[CycleStep] = []
    direction = start_direction
    lots = base_lots
    balance = 0.0
    capital_ok = True
    peak_intra_loss = 0.0
    whipsaw_count = 0
    total_fees = 0.0

    # Entry price depends on starting direction
    initial_price = lp if direction == "LONG" else sp

    # Check if even the first position can be opened
    if (lots * initial_price) / leverage > capital:
        return steps, start_idx, 0.0, False, 0.0, 0, 0.0

    # Step 1: open in start_direction
    steps.append(CycleStep(
        direction=direction, trigger_price=initial_price, lots=lots,
        pnl=0.0, candle_idx=start_idx,
        timestamp=candles[start_idx].timestamp,
    ))

    for idx in range(start_idx, len(candles)):
        candle = candles[idx]

        if direction == "LONG":
            # Check TP first (TL)
            if candle.high >= tl:
                fee = lots * tl * fee_rate
                slip = lots * tl * slippage_pct
                total_fees += fee
                exit_pnl = (lots * d * leverage) - fee - slip
                balance += exit_pnl
                steps.append(CycleStep(
                    direction="EXIT_LONG", trigger_price=tl, lots=0,
                    pnl=round(exit_pnl, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

            # Check SL (SP)
            if candle.low <= sp:
                fee = lots * sp * fee_rate
                slip = lots * sp * slippage_pct
                total_fees += fee
                loss = -(lots * c * leverage) - fee - slip
                balance += loss
                whipsaw_count += 1
                if balance < peak_intra_loss:
                    peak_intra_loss = balance
                # Correct lot formula: first inversion = base_lots × (x-1) = base_lots × 0.5
                # Subsequent inversions = previous_lots × x = previous_lots × 1.5
                next_lots = round(base_lots * 0.5, 6) if whipsaw_count == 1 else round(lots * 1.5, 6)
                margin_needed = (next_lots * sp) / leverage

                if margin_needed > capital or abs(balance) > capital:
                    return steps, idx, round(balance, 4), False, round(peak_intra_loss, 4), whipsaw_count, round(total_fees, 4)

                steps.append(CycleStep(
                    direction="SHORT", trigger_price=sp, lots=next_lots,
                    pnl=round(loss, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                lots = next_lots
                direction = "SHORT"

                # Max whipsaw limit — stop AFTER the inversion step is logged
                if max_whipsaws > 0 and whipsaw_count >= max_whipsaws:
                    steps.append(CycleStep(
                        direction="STOPPED", trigger_price=sp, lots=0,
                        pnl=0.0, candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

        else:  # SHORT
            # Check TP first (TS)
            if candle.low <= ts_level:
                fee = lots * ts_level * fee_rate
                slip = lots * ts_level * slippage_pct
                total_fees += fee
                exit_pnl = (lots * d * leverage) - fee - slip
                balance += exit_pnl
                steps.append(CycleStep(
                    direction="EXIT_SHORT", trigger_price=ts_level, lots=0,
                    pnl=round(exit_pnl, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

            # Check SL (LP)
            if candle.high >= lp:
                fee = lots * lp * fee_rate
                slip = lots * lp * slippage_pct
                total_fees += fee
                loss = -(lots * c * leverage) - fee - slip
                balance += loss
                whipsaw_count += 1
                if balance < peak_intra_loss:
                    peak_intra_loss = balance
                # Correct lot formula: first inversion = base_lots × (x-1) = base_lots × 0.5
                # Subsequent inversions = previous_lots × x = previous_lots × 1.5
                next_lots = round(base_lots * 0.5, 6) if whipsaw_count == 1 else round(lots * 1.5, 6)
                margin_needed = (next_lots * lp) / leverage

                if margin_needed > capital or abs(balance) > capital:
                    return steps, idx, round(balance, 4), False, round(peak_intra_loss, 4), whipsaw_count, round(total_fees, 4)

                steps.append(CycleStep(
                    direction="LONG", trigger_price=lp, lots=next_lots,
                    pnl=round(loss, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                lots = next_lots
                direction = "LONG"

                # Max whipsaw limit — stop AFTER the inversion step is logged
                if max_whipsaws > 0 and whipsaw_count >= max_whipsaws:
                    steps.append(CycleStep(
                        direction="STOPPED", trigger_price=lp, lots=0,
                        pnl=0.0, candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

    # Ran out of candles without resolving
    return steps, len(candles) - 1, round(balance, 4), capital_ok, round(peak_intra_loss, 4), 0, round(total_fees, 4)


def run_backtest(
    symbol: str,
    trading_candles: List[Candle],
    atr_candles: List[Candle],
    atr_period: int = 14,
    base_lots: float = 1.0,
    leverage: float = 10.0,
    capital: float = 1000.0,
    fee_rate: float = 0.0005,
    buffer: float = 0.8,
    slippage_pct: float = 0.0,
    reward_ratio: float = 2.5,
    max_whipsaws: int = 0,
    atr_guard: bool = True,
    atr_guard_multiplier: float = 1.0,
) -> BacktestResult:
    from chv_engine import calculate_params

    cycles: List[Cycle] = []
    equity = 0.0
    equity_curve = [0.0]
    cycle_pnls = []
    peak_equity = 0.0
    max_drawdown = 0.0
    max_dd_cycle = 0

    i = atr_period + 2
    cycle_num = 0
    total_fees = 0.0
    # running_capital grows/shrinks with accumulated P&L so each new cycle
    # uses the actual available balance for margin and loss-limit checks.
    running_capital = capital
    # First cycle always starts LONG; subsequent cycles continue in the
    # same direction as the previous exit (momentum continuation).
    start_direction = "LONG"

    while i < len(trading_candles) - 1:
        # ATR from macro TF aligned to current position
        aligned = align_atr_candles(trading_candles, atr_candles, i)
        if len(aligned) < atr_period + 1:
            i += 1
            continue

        atr_val = calc_atr(aligned, atr_period)
        if atr_val <= 0:
            i += 1
            continue

        entry_price = trading_candles[i].close
        params = calculate_params(symbol, entry_price, atr_val, buffer, reward_ratio, atr_guard_multiplier)

        # ATR Guard: compare footprint to trading TF ATR (not macro ATR).
        # Macro ATR sizes the bracket; trading TF ATR reflects how much price
        # actually moves per candle right now. If footprint exceeds the trading
        # TF ATR × multiplier, each candle is too small to resolve the cycle →
        # likely whipsaw territory. Skip and wait for volatility to return.
        if atr_guard:
            trading_tf_atr = calc_atr(
                trading_candles[max(0, i - atr_period):i + 1], atr_period
            )
            if trading_tf_atr > 0 and params.footprint > trading_tf_atr * atr_guard_multiplier:
                i += 1
                continue

        steps, end_idx, net_pnl, capital_ok, peak_intra_loss, liq_step, cycle_fees = simulate_cycle_on_candles(
            trading_candles, i,
            params.lp, params.sp, params.tl, params.ts,
            params.c, params.d,
            base_lots, leverage, running_capital, fee_rate, slippage_pct,
            start_direction=start_direction,
            max_whipsaws=max_whipsaws,
        )

        # No steps means the first-position margin check failed — skip this candle.
        if not steps:
            i += 1
            continue

        cycle_num += 1
        whipsaws = max(0, len([s for s in steps if s.direction in ("LONG", "SHORT")]) - 1)

        last_step = steps[-1]
        if "EXIT" in last_step.direction:
            exit_dir = last_step.direction
        elif last_step.direction == "STOPPED":
            exit_dir = "STOPPED"
        else:
            exit_dir = "ABORTED"

        equity += net_pnl
        total_fees += cycle_fees
        running_capital = capital + equity
        equity_curve.append(round(equity, 2))
        cycle_pnls.append(net_pnl)

        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_dd_cycle = cycle_num

        duration = end_idx - i + 1
        cycle = Cycle(
            cycle_num=cycle_num,
            entry_price=entry_price,
            atr_at_entry=round(atr_val, 6),
            c=params.c,
            d=params.d,
            lp=params.lp,
            sp=params.sp,
            tl=params.tl,
            ts=params.ts,
            base_lots=base_lots,
            steps=steps,
            whipsaws=whipsaws,
            net_pnl=net_pnl,
            start_idx=i,
            end_idx=end_idx,
            start_ts=trading_candles[i].timestamp,
            end_ts=trading_candles[end_idx].timestamp,
            duration_candles=duration,
            exit_direction=exit_dir,
            capital_ok=capital_ok,
            peak_intra_loss=peak_intra_loss,
        )
        cycles.append(cycle)

        # Stop if the account is genuinely blown (liquidation cycle is now recorded above).
        if not capital_ok or running_capital <= 0:
            break

        # Momentum continuation: next cycle starts in same direction as this exit.
        # ABORTED cycles keep the same direction as before.
        # STOPPED cycles reset to LONG (neutral re-entry after deliberate stop).
        if exit_dir == "EXIT_LONG":
            start_direction = "LONG"
        elif exit_dir == "EXIT_SHORT":
            start_direction = "SHORT"
        elif exit_dir == "STOPPED":
            start_direction = "LONG"

        i = end_idx + 1

    liquidated = any(not c.capital_ok for c in cycles)
    liq_cycle_obj = next((c for c in cycles if not c.capital_ok), None)

    winning = sum(1 for c in cycles if "EXIT" in c.exit_direction and c.net_pnl > 0)
    aborted = sum(1 for c in cycles if not c.capital_ok)
    all_whipsaws = [c.whipsaws for c in cycles]
    durations = [c.duration_candles for c in cycles]

    intra_losses = [c.peak_intra_loss for c in cycles if c.peak_intra_loss < 0]
    worst_intra = min(intra_losses) if intra_losses else 0.0
    worst_intra_cycle = next(
        (c.cycle_num for c in cycles if c.peak_intra_loss == worst_intra), 0
    )

    return BacktestResult(
        symbol=symbol,
        trading_tf="",
        atr_tf="",
        period_start=trading_candles[0].timestamp if trading_candles else 0,
        period_end=trading_candles[-1].timestamp if trading_candles else 0,
        total_cycles=len(cycles),
        winning_cycles=winning,
        aborted_cycles=aborted,
        total_net_pnl=round(equity, 2),
        total_fees=round(total_fees, 2),
        avg_whipsaws=round(statistics.mean(all_whipsaws), 2) if all_whipsaws else 0,
        max_whipsaws=max(all_whipsaws) if all_whipsaws else 0,
        avg_duration_candles=round(statistics.mean(durations), 1) if durations else 0,
        peak_equity=round(peak_equity, 2),
        max_drawdown=round(max_drawdown, 2),
        max_drawdown_cycle=max_dd_cycle,
        worst_intra_loss=round(worst_intra, 2),
        worst_intra_loss_cycle=worst_intra_cycle,
        liquidated=liquidated,
        liquidation_cycle=liq_cycle_obj.cycle_num if liq_cycle_obj else 0,
        liquidation_step=len([s for s in liq_cycle_obj.steps if s.direction in ("LONG","SHORT")]) if liq_cycle_obj else 0,
        liquidation_loss=liq_cycle_obj.net_pnl if liq_cycle_obj else 0.0,
        data_exhausted=(not liquidated),
        equity_curve=equity_curve,
        cycle_pnls=cycle_pnls,
        cycles=cycles,
    )
