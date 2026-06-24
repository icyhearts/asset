#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


SGE_DAILY_URL = "https://www.sge.com.cn/sjzx/quotation_daily_new"
CONTRACT_ID = "Au99.99"
MAX_QUERY_DAYS = 31
REQUEST_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class GoldRow:
    trade_date: date
    low: Decimal
    high: Decimal
    open: Decimal
    close: Decimal


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}, expected YYYY-MM-DD"
        ) from exc


def parse_decimal(value) -> Decimal:
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "NaN", "None"}:
        raise ValueError(f"invalid price value: {value!r}")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid price value: {value!r}") from exc


def format_price(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def date_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=MAX_QUERY_DAYS - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def request_html(
    session: requests.Session, start: date, end: date, contract: str
) -> str:
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "inst_ids": contract,
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(
                SGE_DAILY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    assert last_error is not None
    raise RuntimeError(
        f"failed to query SGE daily data for {start}..{end}: {last_error}"
    )


def find_daily_table(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()

    required_columns = {"日期", "合约", "开盘价", "最高价", "最低价", "收盘价"}
    for table in tables:
        table = table.copy()
        table.columns = [str(column).strip() for column in table.columns]
        if required_columns.issubset(set(table.columns)):
            return table
    return pd.DataFrame()


def parse_rows_from_html(html: str, contract: str) -> list[GoldRow]:
    table = find_daily_table(html)
    if table.empty:
        return []

    rows: list[GoldRow] = []
    for _, row in table.iterrows():
        if str(row["合约"]).strip() != contract:
            continue
        try:
            trade_date = date.fromisoformat(str(row["日期"]).strip())
            rows.append(
                GoldRow(
                    trade_date=trade_date,
                    low=parse_decimal(row["最低价"]),
                    high=parse_decimal(row["最高价"]),
                    open=parse_decimal(row["开盘价"]),
                    close=parse_decimal(row["收盘价"]),
                )
            )
        except Exception as exc:
            raise RuntimeError(f"failed to parse SGE row: {row.to_dict()}") from exc
    return rows


def query_gold_rows(start: date, end: date) -> list[GoldRow]:
    session = requests.Session()
    rows_by_date: dict[date, GoldRow] = {}
    for chunk_start, chunk_end in date_chunks(start, end):
        html = request_html(session, chunk_start, chunk_end, CONTRACT_ID)
        for row in parse_rows_from_html(html, CONTRACT_ID):
            if start <= row.trade_date <= end:
                rows_by_date[row.trade_date] = row
    return [rows_by_date[key] for key in sorted(rows_by_date)]


def save_csv(rows: list[GoldRow], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        for row in rows:
            writer.writerow(
                [
                    row.trade_date.isoformat(),
                    format_price(row.low),
                    format_price(row.high),
                    format_price(row.open),
                    format_price(row.close),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Shanghai Gold Exchange Au99.99 daily OHLC data."
    )
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
    parser.add_argument("--save-path", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        print("--start-date must be <= --end-date", file=sys.stderr)
        return 2

    rows = query_gold_rows(args.start_date, args.end_date)
    save_csv(rows, args.save_path)
    print(f"saved {len(rows)} rows to {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
