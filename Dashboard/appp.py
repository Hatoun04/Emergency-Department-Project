import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ER Wait Time Dashboard", layout="wide")
st.title("🏥 ER Wait Time Dashboard")

# --- Load data ---
FILE_PATH = "ER Wait Time Dataset.csv"
df = pd.read_csv(FILE_PATH)

# --- Time columns ---
df["Visit Date"] = pd.to_datetime(df["Visit Date"])
if "hour" not in df.columns:
    df["hour"] = df["Visit Date"].dt.hour

# Order day names (if strings)
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
if df["Day of Week"].dtype == "O":
    df["Day of Week"] = pd.Categorical(df["Day of Week"], categories=dow_order, ordered=True)

# Which wait-time column to use
WAIT_COL = "Total Wait Time (min)"

# --- Sidebar filters ---
st.sidebar.header("Filters")
sel_hosp   = st.sidebar.multiselect("Hospital ID", sorted(df["Hospital ID"].unique()), default=[])
sel_region = st.sidebar.multiselect("Region",      sorted(df["Region"].unique()),      default=[])
sel_season = st.sidebar.multiselect("Season",      sorted(df["Season"].unique()),      default=[])
sel_tod    = st.sidebar.multiselect("Time of Day", sorted(df["Time of Day"].unique()), default=[])

date_range = st.sidebar.date_input(
    "Date range",
    [df["Visit Date"].min().date(), df["Visit Date"].max().date()],
)

# Moving average (days)
smooth_w = st.sidebar.slider("Moving average window (days)", 1, 14, 1, 1)

# Apply filters
m = pd.Series(True, index=df.index)
if sel_hosp:   m &= df["Hospital ID"].isin(sel_hosp)
if sel_region: m &= df["Region"].isin(sel_region)
if sel_season: m &= df["Season"].isin(sel_season)
if sel_tod:    m &= df["Time of Day"].isin(sel_tod)
if date_range and len(date_range) == 2:
    s = pd.to_datetime(date_range[0])
    e = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    m &= (df["Visit Date"] >= s) & (df["Visit Date"] < e)

df_f = df[m].copy()

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
mean_wait = df_f[WAIT_COL].mean()
p90_wait  = df_f[WAIT_COL].quantile(0.90)
peak_hr   = int(df_f.groupby("hour")[WAIT_COL].mean().idxmax()) if not df_f.empty else None
peak_day  = str(df_f.groupby("Day of Week")[WAIT_COL].mean().idxmax()) if not df_f.empty else None

c1.metric("Rows", f"{len(df_f):,}")
c2.metric("Avg wait (min)", f"{mean_wait:.1f}" if not np.isnan(mean_wait) else "—")
c3.metric("90th pct (min)", f"{p90_wait:.1f}"  if not np.isnan(p90_wait)  else "—")
c4.metric("Peak hour/day", f"{peak_hr}:00 / {peak_day}" if peak_hr is not None and peak_day is not None else "—")

# --- Time series + moving average ---
st.markdown("### Total Wait Time Over Time")
ts = df_f[["Visit Date", WAIT_COL]].sort_values("Visit Date")
fig_ts = px.line(ts, x="Visit Date", y=WAIT_COL, title=f"{WAIT_COL} Over Time")
if smooth_w > 1 and not ts.empty:
    ts_ma = ts.set_index("Visit Date").rolling(f"{smooth_w}D")[WAIT_COL].mean().reset_index()
    fig_ts.add_trace(go.Scatter(x=ts_ma["Visit Date"], y=ts_ma[WAIT_COL], mode="lines", name=f"MA {smooth_w}d"))
st.plotly_chart(fig_ts, use_container_width=True)

# --- Heatmap (Day of Week x Hour) ---
st.markdown("### Heatmap: Avg wait by Day of Week and Hour")
if not df_f.empty:
    piv = df_f.pivot_table(index="Day of Week", columns="hour", values=WAIT_COL, aggfunc="mean")
    fig_h = px.imshow(piv, aspect="auto", origin="lower", labels=dict(color=f"Avg {WAIT_COL}"))
    st.plotly_chart(fig_h, use_container_width=True)

# --- Distribution ---
st.markdown("### Distribution of Wait Times")
fig_hist = px.histogram(df_f, x=WAIT_COL, nbins=30)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Simple staffing suggestion (rule-of-thumb) ---
st.markdown("### Staffing suggestions (rule-of-thumb)")
thr  = st.slider("Target max average wait (min) before adding staff", 30, 180, 90, 5)
step = st.slider("Add 1 nurse per extra minutes above target (min)",   20, 120, 60, 5)

by_slot = df_f.copy()
by_slot["date_hour"] = df_f["Visit Date"].dt.floor("H")
g = by_slot.groupby("date_hour")[WAIT_COL].mean().rename("avg_wait").reset_index()

def extra_nurses(w):
    return max(0, int(np.ceil((w - thr)/step))) if w > thr else 0

g["extra_nurses"] = g["avg_wait"].apply(extra_nurses)
top = g.sort_values("extra_nurses", ascending=False).head(20)

fig_staff = px.bar(top, x="date_hour", y="extra_nurses", title="Top time windows needing extra nurses")
st.plotly_chart(fig_staff, use_container_width=True)
st.dataframe(top)

# --- Download filtered data ---
st.download_button(
    "Download filtered data (CSV)",
    df_f.to_csv(index=False).encode("utf-8"),
    file_name="filtered_er.csv",
    mime="text/csv"
)
