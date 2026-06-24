#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path


@dataclass(frozen=True)
class PriceRow:
    trade_date: date
    low: Decimal
    high: Decimal
    open: Decimal
    close: Decimal


@dataclass(frozen=True)
class InventoryLot:
    buy_date: date
    buy_price: Decimal


@dataclass(frozen=True)
class BuySignal:
    buy_price: Decimal
    trigger_price: Decimal


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}, expected YYYY-MM-DD"
        ) from exc


def parse_decimal_arg(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value {value!r}") from exc


def parse_decimal(value: str) -> Decimal:
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "NaN", "None"}:
        raise ValueError(f"invalid price value: {value!r}")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid price value: {value!r}") from exc


def format_decimal(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def read_price_rows(read_path: Path) -> list[PriceRow]:
    rows_by_date: dict[date, PriceRow] = {}
    with read_path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.reader(input_file)
        for line_no, fields in enumerate(reader, start=1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) < 5:
                raise ValueError(f"{read_path}:{line_no}: expected 5 columns")
            try:
                trade_date = date.fromisoformat(fields[0].strip())
            except ValueError:
                if line_no == 1 and fields[0].strip().lower() in {"date", "日期"}:
                    continue
                raise
            rows_by_date[trade_date] = PriceRow(
                trade_date=trade_date,
                low=parse_decimal(fields[1]),
                high=parse_decimal(fields[2]),
                open=parse_decimal(fields[3]),
                close=parse_decimal(fields[4]),
            )
    return [rows_by_date[key] for key in sorted(rows_by_date)]


def maybe_sell_one(
    current: PriceRow,
    inventory: list[InventoryLot],
    sell_profit: Decimal,
) -> Decimal:
    candidates = [
        (index, lot)
        for index, lot in enumerate(inventory)
        if current.high - lot.buy_price >= sell_profit
    ]
    if not candidates:
        return Decimal("0")

    index, lot = min(candidates, key=lambda item: item[1].buy_price)
    sell_price = lot.buy_price + sell_profit
    inventory.pop(index)
    print(
        "SELL "
        f"date={current.trade_date.isoformat()} "
        f"sell_price={format_decimal(sell_price)} "
        f"buy_date={lot.buy_date.isoformat()} "
        f"buy_price={format_decimal(lot.buy_price)} "
        f"day_low={format_decimal(current.low)} "
        f"day_high={format_decimal(current.high)} "
        f"profit={format_decimal(sell_profit)}"
    )
    return sell_profit


def get_buy_signal(
    rows: list[PriceRow], index: int, buy_prev_interval: int
) -> BuySignal | None:
    if index < buy_prev_interval:
        return None

    previous_rows = rows[index - buy_prev_interval : index]
    trigger_price = min(row.low for row in previous_rows)
    current = rows[index]

    if current.low > trigger_price:
        return None

    # The trigger price is computed only from previous trading days. If the
    # market opens below it, the open is the first executable daily price.
    buy_price = current.open if current.open <= trigger_price else trigger_price
    return BuySignal(buy_price=buy_price, trigger_price=trigger_price)


def simulate(
    rows: list[PriceRow],
    start_date: date,
    end_date: date,
    buy_prev_interval: int,
    sell_profit: Decimal,
) -> Decimal:
    inventory: list[InventoryLot] = []
    total_profit = Decimal("0")

    for index, current in enumerate(rows):
        if current.trade_date < start_date or current.trade_date > end_date:
            continue

        total_profit += maybe_sell_one(current, inventory, sell_profit)

        buy_signal = get_buy_signal(rows, index, buy_prev_interval)
        if buy_signal is not None:
            inventory.append(InventoryLot(current.trade_date, buy_signal.buy_price))
            print(
                "BUY "
                f"date={current.trade_date.isoformat()} "
                f"buy_price={format_decimal(buy_signal.buy_price)} "
                f"trigger_price={format_decimal(buy_signal.trigger_price)} "
                f"day_low={format_decimal(current.low)} "
                f"day_high={format_decimal(current.high)}"
            )

    print(f"TOTAL_PROFIT {format_decimal(total_profit)}")
    print(f"INVENTORY_LEFT {len(inventory)}")
    return total_profit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate Shanghai gold trading.")
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--read-path", required=True, type=Path)
    parser.add_argument("--buy-prev-interval", required=True, type=int)
    parser.add_argument("--sell-profit", required=True, type=parse_decimal_arg)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        print("--start-date must be <= --end-date", file=sys.stderr)
        return 2
    if args.buy_prev_interval <= 0:
        print("--buy-prev-interval must be > 0", file=sys.stderr)
        return 2
    if args.sell_profit <= 0:
        print("--sell-profit must be > 0", file=sys.stderr)
        return 2

    rows = read_price_rows(args.read_path)
    if not rows:
        print(f"no rows found in {args.read_path}", file=sys.stderr)
        return 1

    simulate(
        rows,
        args.start_date,
        args.end_date,
        args.buy_prev_interval,
        args.sell_profit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
