"""CHV CoinM Simulator — historical backtest engine (inverse perpetual)."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math
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
    min_notional_rejected: bool   # True if a cycle was stopped by min notional check
    min_notional_cycle: int
    min_notional_price: float
    min_notional_inv_notional: float
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
    """Return ATR (macro) candles that are FULLY CLOSED as of the current trading
    candle's CLOSE — matching the live bot, whose get_ohlcv drops the currently
    forming candle (raw[:-1]). The macro candle still forming during this trading
    candle is excluded, removing the look-ahead from its (historically-complete)
    full range. Decision is made at the trading candle's close, so a macro candle
    is usable only once its own close has passed by then."""
    # Trading candle's close time = next trading candle's open (else derive from spacing)
    if trading_idx + 1 < len(trading_candles):
        t_close = trading_candles[trading_idx + 1].timestamp
    elif len(trading_candles) >= 2:
        t_close = trading_candles[trading_idx].timestamp + (
            trading_candles[1].timestamp - trading_candles[0].timestamp)
    else:
        t_close = trading_candles[trading_idx].timestamp
    if len(atr_candles) >= 2:
        atr_interval = atr_candles[1].timestamp - atr_candles[0].timestamp
        return [c for c in atr_candles if c.timestamp + atr_interval <= t_close]
    return [c for c in atr_candles if c.timestamp < t_close]


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
    fee_rate_maker: float = 0.0002,
    slippage_pct: float = 0.0,
    start_direction: str = "LONG",
    max_whipsaws: int = 0,
    sl_mode: str = 'wick',
    face_value: float = 100.0,  # USD per contract: 100 for BTC, 10 for others
) -> tuple[List[CycleStep], int, float, bool, float, int, float]:
    """
    Simulate one CHV cycle on CoinM (inverse perpetual) contracts.

    P&L formulas (all in USD, settled in base coin):
      LONG  TP: contracts × face_value × d / lp
      LONG  SL: contracts × face_value × c / lp  (loss)
      SHORT TP: contracts × face_value × d / sp
      SHORT SL: contracts × face_value × c / sp  (loss)

    Fee: contracts × face_value × fee_rate  (notional-based, no price)
    Margin: contracts × face_value / leverage  (price-independent)

    Returns (steps, end_idx, net_pnl, capital_ok, peak_intra_loss, liq_step, total_fees).
    """
    steps: List[CycleStep] = []
    direction = start_direction
    lots = base_lots
    balance = 0.0
    capital_ok = True
    peak_intra_loss = 0.0
    whipsaw_count = 0
    total_fees = 0.0

    initial_price = lp if direction == "LONG" else sp
    entry_px = initial_price   # ACTUAL fill price of the current open leg (not the bracket)

    # capital is in BASE COIN (BTC/SOL). CoinM initial margin in base coin =
    # (contracts × face_value) / (leverage × price). Can't open if it exceeds capital.
    if (lots * face_value) / (leverage * initial_price) > capital:
        return steps, start_idx, 0.0, False, 0.0, 0, 0.0

    # Open entry position — fee in base coin = notional_USD × rate / fill_price
    entry_fee  = lots * face_value * fee_rate / initial_price
    entry_slip = lots * face_value * slippage_pct / initial_price
    total_fees += entry_fee
    balance    -= (entry_fee + entry_slip)
    steps.append(CycleStep(
        direction=direction, trigger_price=initial_price, lots=lots,
        pnl=round(-(entry_fee + entry_slip), 4), candle_idx=start_idx,
        timestamp=candles[start_idx].timestamp,
    ))

    # Inverse-perp realized P&L (base coin) from a leg's ACTUAL entry to its exit.
    # Using the real entry (not the bracket / d-shortcut) is what makes close-mode
    # overshoot fills book correctly.
    def _pnl(dir_, lots_, ent, ex):
        if dir_ == "LONG":
            return lots_ * face_value * (1.0 / ent - 1.0 / ex)
        return lots_ * face_value * (1.0 / ex - 1.0 / ent)

    # Start from start_idx + 1: LP = candle[start_idx].close, so that candle's
    # high/low happened BEFORE the close. Orders are placed after close; the
    # first candle eligible for TP/SL is the next one.
    for idx in range(start_idx + 1, len(candles)):
        candle = candles[idx]

        if direction == "LONG":
            # TP at TL — resting limit (maker), fills at the limit price tl
            if candle.high >= tl:
                fee  = lots * face_value * fee_rate_maker / tl
                slip = lots * face_value * slippage_pct / tl
                total_fees += fee
                # P&L from this leg's ACTUAL entry (not the d-shortcut)
                exit_pnl = _pnl("LONG", lots, entry_px, tl) - fee - slip
                balance += exit_pnl
                steps.append(CycleStep(
                    direction="EXIT_LONG", trigger_price=tl, lots=0,
                    pnl=round(exit_pnl, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

            # SL at SP
            sl_trigger = candle.close <= sp if sl_mode == 'close' else candle.low <= sp
            if sl_trigger:
                # Real fill: close-mode market-fills at the candle close (overshoot past SP);
                # wick-mode (stop-market) fills at the bracket, gap-aware.
                sl_fill = candle.close if sl_mode == 'close' else min(candle.open, sp)
                fee  = lots * face_value * fee_rate / sl_fill
                slip = lots * face_value * slippage_pct / sl_fill
                total_fees += fee
                loss = _pnl("LONG", lots, entry_px, sl_fill) - fee - slip   # negative
                balance += loss
                whipsaw_count += 1
                if balance < peak_intra_loss:
                    peak_intra_loss = balance

                next_lots = round(base_lots * 0.5, 6) if whipsaw_count == 1 else round(lots * 1.5, 6)
                # Liquidation check in BASE COIN: realized losses so far reduce equity;
                # can't post the next leg's margin (price-aware) → cycle blows up.
                next_margin = (next_lots * face_value) / (leverage * sl_fill)
                equity = capital + balance   # balance is negative during cascade
                if equity <= 0 or next_margin > equity:
                    return steps, idx, round(balance, 4), False, round(peak_intra_loss, 4), whipsaw_count, round(total_fees, 4)

                # Invert to SHORT. Entry = REAL fill (close mode) or bracket SP (wick — unchanged).
                inv_entry = sl_fill if sl_mode == 'close' else sp
                open_fee  = next_lots * face_value * fee_rate / sl_fill
                open_slip = next_lots * face_value * slippage_pct / sl_fill
                total_fees += open_fee
                balance    -= (open_fee + open_slip)
                if balance < peak_intra_loss:
                    peak_intra_loss = balance

                steps.append(CycleStep(
                    direction="SHORT", trigger_price=sp, lots=next_lots,
                    pnl=round(loss, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                lots = next_lots
                direction = "SHORT"
                entry_px = inv_entry

                if max_whipsaws > 0 and whipsaw_count >= max_whipsaws:
                    steps.append(CycleStep(
                        direction="STOPPED", trigger_price=sp, lots=0,
                        pnl=0.0, candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

                # Instant-TP: inverted SHORT opened already past its TP (entry ≤ TS) → the
                # reduce-only TP limit is immediately marketable → same-candle exit (taker).
                # (Cannot fire in wick mode: inv_entry = SP > TS.)
                if entry_px <= ts_level:
                    fee  = lots * face_value * fee_rate / ts_level
                    slip = lots * face_value * slippage_pct / ts_level
                    total_fees += fee
                    exit_pnl = _pnl("SHORT", lots, entry_px, ts_level) - fee - slip
                    balance += exit_pnl
                    if balance < peak_intra_loss:
                        peak_intra_loss = balance
                    steps.append(CycleStep(
                        direction="EXIT_SHORT", trigger_price=ts_level, lots=0,
                        pnl=round(exit_pnl, 4), candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

        else:  # SHORT
            # TP at TS — resting limit (maker), fills at the limit price ts
            if candle.low <= ts_level:
                fee  = lots * face_value * fee_rate_maker / ts_level
                slip = lots * face_value * slippage_pct / ts_level
                total_fees += fee
                # P&L from this leg's ACTUAL entry (not the d-shortcut)
                exit_pnl = _pnl("SHORT", lots, entry_px, ts_level) - fee - slip
                balance += exit_pnl
                steps.append(CycleStep(
                    direction="EXIT_SHORT", trigger_price=ts_level, lots=0,
                    pnl=round(exit_pnl, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

            # SL at LP
            sl_trigger = candle.close >= lp if sl_mode == 'close' else candle.high >= lp
            if sl_trigger:
                # Real fill: close-mode market-fills at the candle close (overshoot past LP);
                # wick-mode (stop-market) fills at the bracket, gap-aware.
                sl_fill = candle.close if sl_mode == 'close' else max(candle.open, lp)
                fee  = lots * face_value * fee_rate / sl_fill
                slip = lots * face_value * slippage_pct / sl_fill
                total_fees += fee
                loss = _pnl("SHORT", lots, entry_px, sl_fill) - fee - slip   # negative
                balance += loss
                whipsaw_count += 1
                if balance < peak_intra_loss:
                    peak_intra_loss = balance

                next_lots = round(base_lots * 0.5, 6) if whipsaw_count == 1 else round(lots * 1.5, 6)
                # Liquidation check in BASE COIN (price-aware), same as LONG branch.
                next_margin = (next_lots * face_value) / (leverage * sl_fill)
                equity = capital + balance   # balance is negative during cascade
                if equity <= 0 or next_margin > equity:
                    return steps, idx, round(balance, 4), False, round(peak_intra_loss, 4), whipsaw_count, round(total_fees, 4)

                # Invert to LONG. Entry = REAL fill (close mode) or bracket LP (wick — unchanged).
                inv_entry = sl_fill if sl_mode == 'close' else lp
                open_fee  = next_lots * face_value * fee_rate / sl_fill
                open_slip = next_lots * face_value * slippage_pct / sl_fill
                total_fees += open_fee
                balance    -= (open_fee + open_slip)
                if balance < peak_intra_loss:
                    peak_intra_loss = balance

                steps.append(CycleStep(
                    direction="LONG", trigger_price=lp, lots=next_lots,
                    pnl=round(loss, 4), candle_idx=idx,
                    timestamp=candle.timestamp,
                ))
                lots = next_lots
                direction = "LONG"
                entry_px = inv_entry

                if max_whipsaws > 0 and whipsaw_count >= max_whipsaws:
                    steps.append(CycleStep(
                        direction="STOPPED", trigger_price=lp, lots=0,
                        pnl=0.0, candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

                # Instant-TP: inverted LONG opened already past its TP (entry ≥ TL) → the
                # reduce-only TP limit is immediately marketable → same-candle exit (taker).
                # (Cannot fire in wick mode: inv_entry = LP < TL.)
                if entry_px >= tl:
                    fee  = lots * face_value * fee_rate / tl
                    slip = lots * face_value * slippage_pct / tl
                    total_fees += fee
                    exit_pnl = _pnl("LONG", lots, entry_px, tl) - fee - slip
                    balance += exit_pnl
                    if balance < peak_intra_loss:
                        peak_intra_loss = balance
                    steps.append(CycleStep(
                        direction="EXIT_LONG", trigger_price=tl, lots=0,
                        pnl=round(exit_pnl, 4), candle_idx=idx,
                        timestamp=candle.timestamp,
                    ))
                    return steps, idx, round(balance, 4), True, round(peak_intra_loss, 4), 0, round(total_fees, 4)

    return steps, len(candles) - 1, round(balance, 4), capital_ok, round(peak_intra_loss, 4), 0, round(total_fees, 4)


def run_backtest(
    symbol: str,
    trading_candles: List[Candle],
    atr_candles: List[Candle],
    atr_period: int = 14,
    atr_period_2: int = 0,
    base_lots: float = 1.0,
    leverage: float = 10.0,
    capital: float = 1000.0,
    fee_rate: float = 0.0005,
    fee_rate_maker: float = 0.0002,
    buffer: float = 0.8,
    slippage_pct: float = 0.0,
    reward_ratio: float = 2.5,
    max_whipsaws: int = 0,
    atr_guard: bool = True,
    atr_guard_multiplier: float = 1.0,
    min_notional_on: bool = False,
    fixed_margin: float = 0.0,
    fixed_coin_margin: float = 0.0,   # base-coin margin per cycle (price-scaled contracts)
    lot_step: float = 1.0,        # CoinM: integer contracts, so default step = 1
    dual_atr_mode: str = 'min',
    sl_mode: str = 'wick',
    face_value: float = 100.0,    # USD per contract: 100 for BTC, 10 for others
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
    running_capital = capital
    start_direction = "LONG"

    mn_rejected = False
    mn_cycle    = 0
    mn_price    = 0.0
    mn_notional = 0.0

    while i < len(trading_candles) - 1:
        aligned = align_atr_candles(trading_candles, atr_candles, i)
        if len(aligned) < atr_period + 1:
            i += 1
            continue

        atr_val = calc_atr(aligned, atr_period)
        if atr_val <= 0:
            i += 1
            continue

        if atr_period_2 > 0 and len(aligned) >= atr_period_2 + 1:
            atr_val_2 = calc_atr(aligned, atr_period_2)
            if atr_val_2 > 0:
                atr_val = min(atr_val, atr_val_2) if dual_atr_mode == 'min' else max(atr_val, atr_val_2)

        entry_price = trading_candles[i].close
        params = calculate_params(symbol, entry_price, atr_val, buffer, reward_ratio, atr_guard_multiplier)

        # Sizing mode (precedence: fixed-coin > fixed-USD > base lots)
        if fixed_coin_margin > 0 and lot_step > 0:
            # Fixed base-coin margin: commit a constant amount of the coin each
            # cycle. SUI margin = contracts × face / (lev × price), so to hold it
            # constant the contract count scales WITH price.
            raw = fixed_coin_margin * leverage * entry_price / face_value / lot_step
            cycle_lots = round(math.floor(raw) * lot_step, 8)
            if cycle_lots <= 0:
                i += 1
                continue
        elif fixed_margin > 0 and lot_step > 0:
            # Fixed USD margin: CoinM margin = contracts × face / leverage (price-independent)
            raw = fixed_margin * leverage / face_value / lot_step
            cycle_lots = round(math.floor(raw) * lot_step, 8)
            if cycle_lots <= 0:
                i += 1
                continue
        else:
            cycle_lots = base_lots

        # Min notional check: WS1 inversion contracts × face_value must meet minimum
        if min_notional_on:
            inv_lots = round(cycle_lots * 0.5, 6)
            if inv_lots * face_value < face_value:  # less than 1 contract
                mn_rejected = True
                mn_cycle    = cycle_num + 1
                mn_price    = entry_price
                mn_notional = round(inv_lots * face_value, 4)
                break

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
            cycle_lots, leverage, running_capital, fee_rate, fee_rate_maker, slippage_pct,
            start_direction=start_direction,
            max_whipsaws=max_whipsaws,
            sl_mode=sl_mode,
            face_value=face_value,
        )

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
            base_lots=cycle_lots,
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

        if not capital_ok or running_capital <= 0:
            break

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
        data_exhausted=(not liquidated and not mn_rejected),
        min_notional_rejected=mn_rejected,
        min_notional_cycle=mn_cycle,
        min_notional_price=mn_price,
        min_notional_inv_notional=mn_notional,
        equity_curve=equity_curve,
        cycle_pnls=cycle_pnls,
        cycles=cycles,
    )
