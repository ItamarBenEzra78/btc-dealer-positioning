"""
db.py — The persistence layer (SQLAlchemy ORM).

Replaces flat CSV/JSONL files with a real database behind a repository interface.
Uses SQLite for local dev; switch to Postgres/TimescaleDB by setting one env var:

    export DATABASE_URL="postgresql+psycopg://user:pass@host/db"

Nothing else in the codebase changes — that is the point of the repository pattern.

Tables:
    price_bars   daily BTC OHLC + DVOL (the historical market series)
    snapshots    every dealer-positioning snapshot the collector captures

Run:  python3 db.py     # create tables + migrate existing CSV/JSONL in
"""

import json
import os
from typing import Optional, List, Dict

import pandas as pd
from sqlalchemy import String, Float, Integer, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/app.db")
engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass


class PriceBar(Base):
    __tablename__ = "price_bars"
    date: Mapped[str] = mapped_column(String, primary_key=True)   # ISO date
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)
    dvol: Mapped[Optional[float]] = mapped_column(Float)


class Snapshot(Base):
    __tablename__ = "snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asof: Mapped[str] = mapped_column(String, index=True)
    currency: Mapped[str] = mapped_column(String, index=True)
    spot: Mapped[float] = mapped_column(Float)
    net_gex: Mapped[float] = mapped_column(Float)
    regime: Mapped[str] = mapped_column(String)
    call_wall: Mapped[Optional[float]] = mapped_column(Float)
    put_wall: Mapped[Optional[float]] = mapped_column(Float)
    zero_gamma: Mapped[Optional[float]] = mapped_column(Float)
    max_pain_agg: Mapped[Optional[float]] = mapped_column(Float)
    max_pain_near: Mapped[Optional[float]] = mapped_column(Float)
    near_dte: Mapped[Optional[float]] = mapped_column(Float)
    a1: Mapped[Optional[float]] = mapped_column(Float)
    a2: Mapped[Optional[float]] = mapped_column(Float)
    p1: Mapped[Optional[float]] = mapped_column(Float)
    n1: Mapped[Optional[float]] = mapped_column(Float)


def init_db():
    Base.metadata.create_all(engine)


# ---- repository ---------------------------------------------------------
def upsert_price_bars(df: pd.DataFrame):
    with Session(engine) as s:
        for _, r in df.iterrows():
            d = r["date"]
            key = str(d.date()) if hasattr(d, "date") else str(d)[:10]
            s.merge(PriceBar(date=key, open=_f(r.get("open")), high=_f(r.get("high")),
                             low=_f(r.get("low")), close=_f(r.get("close")),
                             volume=_f(r.get("volume")), dvol=_f(r.get("dvol"))))
        s.commit()


def get_price_bars() -> pd.DataFrame:
    with Session(engine) as s:
        rows = s.scalars(select(PriceBar).order_by(PriceBar.date)).all()
    df = pd.DataFrame([{"date": r.date, "open": r.open, "high": r.high, "low": r.low,
                        "close": r.close, "volume": r.volume, "dvol": r.dvol} for r in rows])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def save_snapshot(snap: dict):
    z = snap.get("zones", {})
    with Session(engine) as s:
        s.add(Snapshot(
            asof=str(snap.get("captured_at") or snap.get("asof")),
            currency=snap["currency"], spot=snap["spot"], net_gex=snap["net_gex"],
            regime=snap["regime"], call_wall=_f(snap.get("call_wall")),
            put_wall=_f(snap.get("put_wall")), zero_gamma=_f(snap.get("zero_gamma")),
            max_pain_agg=_f(snap.get("max_pain_agg")), max_pain_near=_f(snap.get("max_pain_near")),
            near_dte=_f(snap.get("near_dte")), a1=_f(z.get("A1")), a2=_f(z.get("A2")),
            p1=_f(z.get("P1")), n1=_f(z.get("N1"))))
        s.commit()


def get_snapshots(currency="BTC", limit=200) -> List[Dict]:
    with Session(engine) as s:
        rows = s.scalars(select(Snapshot).where(Snapshot.currency == currency)
                         .order_by(Snapshot.id.desc()).limit(limit)).all()
    return [{"asof": r.asof, "spot": r.spot, "net_gex": r.net_gex, "regime": r.regime,
             "call_wall": r.call_wall, "put_wall": r.put_wall, "zero_gamma": r.zero_gamma,
             "max_pain_near": r.max_pain_near} for r in rows]


def _f(x):
    try:
        return float(x) if x is not None and pd.notna(x) else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    init_db()
    print(f"DB ready at {DATABASE_URL}")

    # migrate existing files in (idempotent)
    try:
        md = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
        upsert_price_bars(md)
        print(f"  migrated {len(md)} price bars")
    except FileNotFoundError:
        print("  no market_daily.csv to migrate")

    n = 0
    try:
        for line in open("data/snapshots.jsonl"):
            if line.strip():
                save_snapshot(json.loads(line)); n += 1
        print(f"  migrated {n} snapshots")
    except FileNotFoundError:
        print("  no snapshots.jsonl to migrate")

    print(f"  price_bars in DB: {len(get_price_bars())}")
    print(f"  snapshots in DB : {len(get_snapshots())}")
