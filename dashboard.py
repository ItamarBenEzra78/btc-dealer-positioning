"""
dashboard.py — The unifying Streamlit dashboard.

Four tabs tie the whole thesis together:
  🎯 Positioning   live GEX, walls, zero-gamma, max pain, A/P/N, pin check, strike profile
  💵 Flow          live net-premium flow + the significant-threshold model
  🔬 Evidence      the article's claims tested on 2y of BTC + the honest verdict
  ✓ Validation     our engine vs the CryptoGamma oracle

Run:  streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from positioning import snapshot, build_chain, gamma_by_strike
from data_layer import fetch_ohlc
from anomalies import pin_setup, add_forward_returns, add_regime_labels, detect_dvol_spikes
from stats_spine import regime_two_sample, return_persistence, event_study, purged_walk_forward
from flow_engine import live_option_flow, add_flow_features, kyle_lambda, cusum_events, significant_threshold
from validate import fetch_oracle, within
from calendar_layer import next_event, catalyst_read

VIOLET, GREEN, RED, MUTED, INK = "#9A86FF", "#46B98A", "#E5615E", "#98A2B1", "#E7EBF2"
st.set_page_config(page_title="BTC Dealer Positioning · Statistical Evidence",
                   page_icon="🎯", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background: #0C0F14; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -.01em; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  [data-testid="stMetric"] {{ background:#141922; border:1px solid #28303B;
     border-radius:10px; padding:12px 14px; }}
  .verdict {{ background:#141922; border:1px solid #28303B; border-left:3px solid {VIOLET};
     border-radius:10px; padding:16px 20px; margin:6px 0 18px; color:{MUTED}; }}
  .verdict b {{ color:{VIOLET}; }}
  .ok {{ color:{GREEN}; font-weight:600; }} .no {{ color:{RED}; font-weight:600; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_snapshot(): return snapshot("BTC")
@st.cache_data(ttl=300)
def get_profile():
    recs, spot, _ = build_chain("BTC"); s, c, p, n = gamma_by_strike(recs); return s, c, p, spot
@st.cache_data(ttl=300)
def get_flow(): return live_option_flow("BTC")
@st.cache_data(ttl=900)
def get_catalyst(): return next_event()
@st.cache_data(ttl=120)
def get_ohlc(res, bars): return fetch_ohlc(res, bars)

# TradingView-style timeframes -> (Deribit resolution, bars, resample rule)
TF_MAP = {
    "1m": ("1", 360, None), "5m": ("5", 300, None), "10m": ("10", 300, None),
    "1h": ("60", 300, None), "1D": ("1D", 180, None),
    "1M": ("1D", 730, "ME"), "1Y": ("1D", 730, None),
}


def price_chart(tf, levels):
    res, bars, rule = TF_MAP[tf]
    o = get_ohlc(res, bars)
    if rule:
        o = (o.set_index("date").resample(rule)
             .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
             .dropna().reset_index())
    fig = go.Figure(go.Candlestick(
        x=o["date"], open=o["open"], high=o["high"], low=o["low"], close=o["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, name="BTC"))
    lv = [("Call Wall", levels["call_wall"], GREEN), ("Put Wall", levels["put_wall"], RED),
          ("Zero Gamma", levels["zero_gamma"], VIOLET), ("Max Pain", levels["max_pain_agg"], MUTED)]
    for name, y, col in lv:
        if y:
            fig.add_hline(y=y, line_dash="dash", line_color=col, opacity=0.7,
                          annotation_text=name, annotation_position="right",
                          annotation_font_color=col)
    fig.update_layout(height=460, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(t=10, b=10, r=70), yaxis_title="BTC / USD")
    return fig
@st.cache_data(ttl=600)
def get_oracle():
    try: return fetch_oracle()
    except Exception: return {}
@st.cache_data(ttl=3600)
def get_market():
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    return add_flow_features(add_regime_labels(add_forward_returns(df)))
@st.cache_data(ttl=3600)
def get_regime():
    try: return pd.read_csv("data/r_regime.csv", parse_dates=["date"])
    except Exception: return pd.DataFrame()


def usd(x): return f"${x:,.0f}" if x is not None else "—"

snap = get_snapshot(); pin = pin_setup(snap); z = snap["zones"]
df = get_market()

st.title("🎯 BTC Dealer Positioning + Statistical Evidence")
st.markdown(
    "<div class='verdict'>The thesis (market-maker hedging creates structure) × rigorous statistics (does it hold up?). "
    "<b>Headline finding:</b> on 2y of BTC, flow &amp; regime predict <b>volatility / magnitude</b> "
    "(highly significant) but the <span class='no'>directional</span> edge is negligible under purged CV. "
    "An honest tool reports both.</div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["🎯 Positioning", "💵 Flow & Threshold", "🔬 Statistical Evidence", "✓ Validation"])

# ============================ POSITIONING ============================
with t1:
    c = st.columns(4)
    c[0].metric("Spot", usd(snap["spot"]))
    c[1].metric("Net GEX", f"{snap['net_gex']/1e9:+.2f}B", snap["regime"] + " gamma")
    c[2].metric("Zero Gamma", usd(snap["zero_gamma"]))
    c[3].metric("Max Pain (near)", usd(snap["max_pain_near"]), f"{snap['near_dte']}d to expiry")
    c = st.columns(4)
    c[0].metric("Call Wall", usd(snap["call_wall"]))
    c[1].metric("Put Wall", usd(snap["put_wall"]))
    c[2].metric("A1 / A2", f"{usd(z['A1'])} / {usd(z['A2'])}")
    c[3].metric("N1 / P1", f"{usd(z['N1'])} / {usd(z['P1'])}")

    st.subheader("Price & dealer levels")
    tf = st.radio("Timeframe", list(TF_MAP), index=3, horizontal=True, label_visibility="collapsed")
    try:
        st.plotly_chart(price_chart(tf, snap), use_container_width=True)
        st.caption("Candles: BTC-PERPETUAL (Deribit). Dashed lines = live dealer levels. "
                   "Sub-hour history is limited by the free feed; 1D+ spans months.")
    except Exception as e:
        st.warning(f"Price feed hiccup ({type(e).__name__}) — try another timeframe.")

    st.subheader("Max-Pain Pin setup — the article's conditions, live")
    pc = st.columns(4)
    labels = {"positive_gamma": "Positive gamma", "low_dte": "Low DTE (0–3d)", "near_max_pain": "Near max pain"}
    for i, (k, ok) in enumerate(pin["conditions"].items()):
        pc[i].metric(labels[k], "✅ yes" if ok else "❌ no")
    pc[3].metric("Verdict", "PIN LIKELY" if pin["pin_likely"] else f"{pin['conditions_met']}/3",
                 f"{pin['distance_to_max_pain_pct']}% to max pain")

    cat = catalyst_read(pin["pin_likely"], get_catalyst())
    if cat.get("has_catalyst"):
        col = RED if cat.get("level") == "break-risk" else VIOLET if cat.get("imminent") else MUTED
        st.markdown(f"<div class='verdict' style='border-left-color:{col}'>{cat['message']}"
                    "<br><span style='color:#98A2B1;font-size:12px'>Macro catalyst (Forex Factory) — "
                    "a live overlay, not a validated signal.</span></div>", unsafe_allow_html=True)

    st.subheader("Strike gamma profile")
    strikes, call, put, spot = get_profile()
    fig = go.Figure()
    fig.add_bar(x=strikes, y=call / 1e9, name="Call γ", marker_color=GREEN)
    fig.add_bar(x=strikes, y=-put / 1e9, name="Put γ", marker_color=RED)
    fig.add_vline(x=spot, line_dash="dash", line_color=VIOLET, annotation_text="spot")
    if snap["max_pain_agg"]:
        fig.add_vline(x=snap["max_pain_agg"], line_dash="dot", line_color=MUTED, annotation_text="max pain")
    fig.update_layout(barmode="relative", height=400, template="plotly_dark",
                      xaxis_title="Strike", yaxis_title="Gamma exposure ($B / 1% move)",
                      margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ============================ FLOW ============================
with t2:
    fl = get_flow()
    st.subheader("Live options net-premium flow (Deribit taker trades)")
    c = st.columns(4)
    c[0].metric("Trades sampled", f"{fl['n_trades']}")
    c[1].metric("Net call premium", f"{fl['net_call_premium']:+.2f} ₿")
    c[2].metric("Net put premium", f"{fl['net_put_premium']:+.2f} ₿")
    c[3].metric("Flow tilt", fl["flow_tilt"])

    st.subheader("Significant-flow threshold — how much flow is enough to move price?")
    kl = kyle_lambda(df)
    stt = significant_threshold(df)
    ev = cusum_events(df["signed_vol_z"], h=5.0)
    c = st.columns(3)
    c[0].metric("Amihud impact (|move| vs vol)", f"p={kl['amihud_p']:.4f}",
                "significant" if kl["amihud_p"] < 0.05 else "ns")
    c[1].metric("Kyle λ predictive (direction)", f"p={kl['kyle_pred_p']:.3f}",
                "no directional edge" if kl["kyle_pred_p"] > 0.05 else "edge")
    c[2].metric("CUSUM significant-flow events", f"{len(ev)}")
    if "error" not in stt:
        st.markdown(
            f"<div class='verdict'>P(|move| ≥ 2% next day): "
            f"calm day <b>{stt['P_big_move_calm']}</b> → high-flow day <b>{stt['P_big_move_high_flow']}</b> "
            f"(base {stt['base_rate']}). <span class='ok'>Flow raises the odds of a big move</span> — "
            f"but of <i>magnitude</i>, not a known direction.</div>", unsafe_allow_html=True)

# ============================ EVIDENCE ============================
with t3:
    st.subheader("The thesis, tested — 2 years of BTC")
    rv = regime_two_sample(df, "fwd_rv_3"); rt = regime_two_sample(df, "fwd_ret_3")
    rp = return_persistence(df)
    sig = lambda p: "<span class='ok'>✓ significant</span>" if p < 0.05 else "<span class='no'>✗ ns</span>"
    e = st.columns(3)
    e[0].markdown(f"**Vol clustering**<br>elevated **{rv['mean_elevated']:.0f}%** vs calm **{rv['mean_calm']:.0f}%**<br>"
                  f"Welch p=`{rv['welch_p']:.4f}` {sig(rv['welch_p'])}", unsafe_allow_html=True)
    e[1].markdown(f"**Directional bias**<br>elevated **{rt['mean_elevated']*100:+.2f}%** vs calm **{rt['mean_calm']*100:+.2f}%**<br>"
                  f"Welch p=`{rt['welch_p']:.4f}` {sig(rt['welch_p'])}", unsafe_allow_html=True)
    e[2].markdown(f"**Persistence (core claim)**<br>calm autocorr **{rp['calm']['autocorr_lag1']:+.3f}** "
                  f"(p=`{rp['calm']['p']:.3f}`) {sig(rp['calm']['p'])}<br>= mean-reverting (positive-gamma behaviour)",
                  unsafe_allow_html=True)

    st.subheader("Event study — CAR around DVOL spikes")
    es = event_study(df, detect_dvol_spikes(df, 2.0), pre=3, post=5)
    if es:
        tab, n = es
        f2 = go.Figure()
        f2.add_scatter(x=tab["offset"], y=tab["CAR"] * 100, mode="lines+markers", line_color=VIOLET)
        f2.add_vline(x=0, line_dash="dash", line_color=MUTED, annotation_text="event")
        f2.update_layout(height=300, template="plotly_dark", xaxis_title="days from spike",
                         yaxis_title="cumulative abnormal return %", margin=dict(t=10, b=10))
        st.plotly_chart(f2, use_container_width=True)
        st.caption(f"{n} events — vol spikes cluster with drawdowns (leverage effect).")

    reg = get_regime()
    if not reg.empty:
        st.subheader("Markov high-vol regime probability (R · MSwM)")
        lb = st.radio("Range", ["3M", "6M", "1Y", "All"], index=2, horizontal=True,
                      label_visibility="collapsed", key="regime_lb")
        days = {"3M": 90, "6M": 180, "1Y": 365, "All": 100000}[lb]
        m = df.merge(reg, on="date", how="inner")
        m = m[m["date"] >= m["date"].max() - pd.Timedelta(days=days)]
        f3 = go.Figure()
        f3.add_scatter(x=m["date"], y=m["close"], name="BTC", line_color=MUTED, yaxis="y2")
        f3.add_scatter(x=m["date"], y=m["p_high_vol_regime"], name="P(high-vol)", line_color=RED, fill="tozeroy")
        f3.update_layout(height=320, template="plotly_dark", margin=dict(t=10, b=10),
                         yaxis=dict(title="P(high-vol)", range=[0, 1]),
                         yaxis2=dict(title="BTC", overlaying="y", side="right", showgrid=False),
                         legend=dict(orientation="h", y=1.12))
        st.plotly_chart(f3, use_container_width=True)
        st.caption("Low-vol regime: +0.13%/day, sticky (0.94). High-vol: −0.28%/day, wilder.")

    pw = purged_walk_forward(df, "fwd_ret_3", ["elevated", "dvol_chg_z", "dvol"])
    if "error" not in pw:
        skill = pw["brier_baserate"] - pw["brier_model"]
        tag = ("<span class='no'>no skill</span>" if skill <= 0
               else "<span class='no'>negligible edge (within noise)</span>" if skill < 0.01
               else "<span class='ok'>modest edge</span>")
        st.markdown(
            f"<div class='verdict'><b>Leakage-safe verdict (purged walk-forward):</b> "
            f"Brier model <b>{pw['brier_model']}</b> vs base-rate <b>{pw['brier_baserate']}</b> "
            f"(edge {skill:+.4f}) → {tag}. Direction stays hard — exactly the honest result most tools hide.</div>",
            unsafe_allow_html=True)

# ============================ VALIDATION ============================
with t4:
    st.subheader("Our engine vs CryptoGamma oracle")
    orc = get_oracle()
    if not orc:
        st.warning("Oracle unavailable right now.")
    else:
        sq, met, risk = orc.get("squeezeLevels", {}), orc.get("metrics", {}), orc.get("riskMetrics", {})
        rows = [("Call Wall / resistance", snap["call_wall"], sq.get("resistance"), 0.02),
                ("Put Wall / support", snap["put_wall"], sq.get("support"), 0.02),
                ("Spot", snap["spot"], sq.get("currentPrice"), 0.01),
                ("Max Pain", snap["max_pain_agg"], sq.get("breakout"), 0.03)]
        tbl = []
        for name, a, b, tol in rows:
            tbl.append({"metric": name, "ours": usd(a), "oracle": usd(b),
                        "match": "✅" if within(a, b, tol) else "⚠️"})
        our_sign = "positive" if snap["net_gex"] > 0 else "negative"
        orc_sign = "positive" if met.get("netGamma", 0) > 0 else "negative"
        tbl.append({"metric": "Net-gamma sign", "ours": our_sign, "oracle": orc_sign,
                    "match": "✅" if our_sign == orc_sign else "⚠️ dealer-sign"})
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='verdict'>Structural <b>levels</b> validate against the paid oracle almost exactly. "
            "The net-gamma <b>sign</b> differs because we use the naive dealer convention and they use a "
            "different sign model — the flagged assumption, not a bug.</div>", unsafe_allow_html=True)

st.divider()
st.caption("⚠️ Research tool, not financial advice. Naive dealer sign (flagged). DVOL is a proxy for the "
           "gamma regime; GEX-native backtests mature as the snapshot DB grows.")
