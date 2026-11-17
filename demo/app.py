"""Streamlit demo: pre-flight battery energy prediction for a delivery UAV.

Run from this folder:
    streamlit run app.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from energy_models import (
    ARTIFACTS,
    FEATURES,
    LR,
    MODEL_KEYS,
    MODEL_NAMES,
    POLY,
    RF,
    TARGET,
    fit_final_models,
    load_flights,
)

# ---- visual tokens (light theme, see .streamlit/config.toml) -----------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {LR: "#2a78d6", RF: "#eb6834", POLY: "#1baf7a"}  # fixed per model
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PAYLOAD_COLOR = {0: "#86b6ef", 250: "#2a78d6", 500: "#104281"}  # ordinal, one hue
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
PLOT_CONFIG = {"displayModeBar": False}

FACTOR_LABEL = {
    "payload": "Tải trọng (g)",
    "speed": "Tốc độ hành trình (m/s)",
    "altitude": "Độ cao hành trình (m)",
}
RANGE = {"payload": (0, 500), "speed": (4, 12), "altitude": (25, 100)}

st.set_page_config(page_title="UAV Energy Planner", page_icon="🔋", layout="wide")


# ---- data ---------------------------------------------------------------------
@st.cache_data
def get_flights() -> pd.DataFrame:
    return load_flights()


@st.cache_resource
def get_models() -> dict:
    return fit_final_models(load_flights())


@st.cache_data
def get_summary() -> dict:
    return json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))


@st.cache_data
def get_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name)


def style(fig: go.Figure, height: int = 340, legend: bool = True, horizontal: bool = False) -> go.Figure:
    """Shared chart chrome. horizontal=True puts the value grid on the x axis."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48 if legend else 28, b=10),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK2),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, size=13, color=INK)),
    )
    fig.update_xaxes(showgrid=horizontal, gridcolor=GRID, gridwidth=1, linecolor=AXIS, ticks="outside",
                     tickcolor=AXIS, zeroline=False, automargin=True, title_standoff=8)
    fig.update_yaxes(showgrid=not horizontal, gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
                     automargin=True, title_standoff=8)
    fig.update_annotations(font=dict(size=13, color=INK))  # subplot titles
    return fig


def config_label(payload: float, speed: float, altitude: float) -> str:
    return f"{payload:.0f} g · {speed:g} m/s · {altitude:.0f} m"


def fmt_p(p: float, sentence: bool = False) -> str:
    if p < 0.001:
        return "p < 0.001" if sentence else "< 0.001"
    return f"p = {p:.3f}" if sentence else f"{p:.3f}"


missing = [f for f in ["summary.json", "per_flight.csv", "cv_folds.csv"] if not (ARTIFACTS / f).exists()]
if missing:
    st.error(
        "Chưa có kết quả đánh giá mô hình. Hãy chạy `python build_artifacts.py` trong thư mục "
        "`demo` rồi tải lại trang.",
        icon=":material/error:",
    )
    st.stop()

flights = get_flights()
models = get_models()
summary = get_summary()
per_flight = get_csv("per_flight.csv")
folds = get_csv("cv_folds.csv")
has_profiles = (ARTIFACTS / "flight_profiles.csv").exists()
best_model = min(MODEL_NAMES, key=lambda m: summary["holdout"][m]["MAE"])

config_stats = (
    flights.groupby(FEATURES)
    .agg(n=("flight", "size"), mean=(TARGET, "mean"), lo=(TARGET, "min"), hi=(TARGET, "max"))
    .reset_index()
)

# ---- sidebar: the flight plan -------------------------------------------------
with st.sidebar:
    st.header("Kế hoạch chuyến bay")
    # bound to the URL, so a scenario can be bookmarked: ?payload=300&speed=7.5&altitude=60
    payload = st.slider("Tải trọng (g)", 0, 500, 250, step=50, key="payload", bind="query-params")
    speed = st.slider("Tốc độ hành trình (m/s)", 4.0, 12.0, 8.0, step=0.5, key="speed", bind="query-params")
    altitude = st.slider("Độ cao hành trình (m)", 25, 100, 50, step=5, key="altitude", bind="query-params")
    st.caption(
        "Thanh trượt bị giới hạn trong phạm vi dữ liệu đã bay thực. "
        "Dữ liệu có đủ lặp tại tải 0/250/500 g, tốc độ 4/6/8/10/12 m/s, độ cao 25/50/75/100 m."
    )
    st.divider()
    model_name = st.selectbox(
        "Mô hình dự báo", MODEL_NAMES, index=MODEL_NAMES.index(best_model),
        help=f"Mặc định là {best_model}, mô hình có sai số thấp nhất trên các chuyến bay giữ kín.",
    )
    level = st.select_slider(
        "Mức tin cậy của khoảng dự báo",
        options=[0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda v: f"{int(v * 100)}%",
    )
    st.divider()
    st.subheader("Pin")
    capacity = st.number_input(
        "Dung lượng danh định (Wh)", min_value=20.0, max_value=500.0, value=99.9, step=0.1,
        help="Pin tiêu chuẩn TB47D của Matrice 100: 4500 mAh × 22.2 V ≈ 99.9 Wh.",
    )
    reserve = st.slider(
        "Mức dự trữ an toàn (%)", 0, 50, 20, step=5,
        help="Phần dung lượng không dùng tới khi lập kế hoạch.",
    )

# ---- prediction for the current plan -------------------------------------------
plan = pd.DataFrame([{"payload": payload, "speed": speed, "altitude": altitude}])
preds = {name: float(mdl.predict(plan)[0]) for name, mdl in models.items()}
same = flights[(flights.payload == payload) & (flights.speed == speed) & (flights.altitude == altitude)]
seen = len(same) > 0
half = {
    name: summary["intervals"][name]["seen" if seen else "unseen"][f"{level:.2f}"]
    for name in MODEL_NAMES
}
pred, lo, hi = preds[model_name], preds[model_name] - half[model_name], preds[model_name] + half[model_name]
usable = capacity * (1 - reserve / 100)
n_per_battery = math.floor(usable / hi) if hi > 0 else 0

st.title("Dự báo năng lượng pin UAV trước chuyến bay")
st.caption(
    f"DTA301 · RBL Nhóm 6 · DJI Matrice 100, tuyến R1 · {summary['data']['n_flights']} chuyến bay thực "
    "(Rodrigues et al., 2021) · chỉ dùng tham số biết trước khi cất cánh"
)

TABS = ["Dự báo", "What-if", "So sánh mô hình", "Phân tích sai số", "Dữ liệu & giới hạn"]
# ?tab=3 in the URL opens the third tab directly (handy when presenting)
tab_param = st.query_params.get("tab", "1")
tab_index = int(tab_param) - 1 if tab_param.isdigit() and 1 <= int(tab_param) <= len(TABS) else 0
tab_predict, tab_whatif, tab_models, tab_errors, tab_data = st.tabs(TABS, default=TABS[tab_index])

# =============================== TAB 1: predict ================================
with tab_predict:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Năng lượng dự báo cả chuyến", f"{pred:.1f} Wh", border=True,
              help=f"Dự báo của {model_name} cho kế hoạch ở thanh bên.")
    k2.metric(f"Khoảng dự báo {int(level * 100)}%", f"{lo:.1f} – {hi:.1f} Wh", border=True,
              help="Dự báo ± phân vị của sai số tuyệt đối ngoài mẫu (xem phần giải thích bên dưới).")
    k3.metric("So với dung lượng pin", f"{pred / capacity * 100:.0f}%", border=True,
              help=f"Cận trên của khoảng dự báo tương ứng {hi / capacity * 100:.0f}% dung lượng danh định.")
    k4.metric("Số chuyến mỗi pin (ước tính)", f"{n_per_battery}", border=True,
              help=f"Phần dùng được {usable:.1f} Wh sau khi trừ {reserve}% dự trữ, chia cho cận trên {hi:.1f} Wh.")

    if seen:
        measured = (
            f"năng lượng đo được {same[TARGET].iloc[0]:.1f} Wh"
            if len(same) == 1
            else f"năng lượng đo được {same[TARGET].min():.1f} – {same[TARGET].max():.1f} Wh, "
                 f"trung bình {same[TARGET].mean():.1f} Wh"
        )
        st.success(
            f"Cấu hình này đã được bay thực {len(same)} lần: {measured}.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            "Cấu hình nội suy: chưa có chuyến bay thực nào với đúng bộ tham số này. Khoảng dự báo "
            "đang dùng sai số đo trên các cấu hình mô hình chưa từng thấy khi huấn luyện.",
            icon=":material/warning:",
        )

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Ba mô hình so với số đo thực")
        rows = ["Chuyến bay thực", POLY, RF, LR] if seen else [POLY, RF, LR]
        fig = go.Figure()
        for name in MODEL_NAMES:
            fig.add_trace(go.Scatter(
                x=[preds[name]], y=[name], mode="markers+text", name=name,
                marker=dict(size=14, color=MODEL_COLOR[name], line=dict(color=SURFACE, width=2)),
                error_x=dict(type="constant", value=half[name], color=MODEL_COLOR[name], thickness=2, width=7),
                text=[f"{preds[name]:.1f} Wh"], textposition="top center", textfont=dict(color=INK, size=13),
                hovertemplate=f"{name}<br>Dự báo %{{x:.2f}} Wh<br>± {half[name]:.2f} Wh<extra></extra>",
            ))
        if seen:
            fig.add_trace(go.Scatter(
                x=same[TARGET], y=["Chuyến bay thực"] * len(same), mode="markers", name="Chuyến bay thực",
                marker=dict(size=12, symbol="diamond", color=INK2, line=dict(color=SURFACE, width=2)),
                customdata=same.flight,
                hovertemplate="Chuyến #%{customdata}<br>Đo được %{x:.2f} Wh<extra></extra>",
            ))
        style(fig, height=310 if seen else 260, legend=False, horizontal=True)
        fig.update_yaxes(categoryorder="array", categoryarray=rows, range=[-0.6, len(rows) - 0.3])
        fig.update_xaxes(title_text="Năng lượng cả chuyến (Wh)")
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
        st.caption(
            f"Thanh ngang là khoảng dự báo {int(level * 100)}% của từng mô hình. "
            + ("Hình thoi xám là các chuyến bay thực cùng cấu hình." if seen else "")
        )

    with right:
        if seen:
            st.subheader("Chuyến bay thực cùng cấu hình")
            ref = same
        else:
            st.subheader("Cấu hình đã bay gần nhất")
            near = config_stats.assign(
                dist=np.sqrt(
                    ((config_stats.payload - payload) / 500) ** 2
                    + ((config_stats.speed - speed) / 8) ** 2
                    + ((config_stats.altitude - altitude) / 75) ** 2
                )
            ).nsmallest(4, "dist")
            near_tbl = near.assign(
                pred=models[model_name].predict(near[FEATURES]),
                config=[config_label(p, s, a) for p, s, a in near[FEATURES].itertuples(index=False)],
            )
            st.dataframe(
                near_tbl[["config", "n", "mean", "pred"]],
                hide_index=True,
                column_config={
                    "config": "Cấu hình",
                    "n": st.column_config.NumberColumn("Số chuyến", format="%d"),
                    "mean": st.column_config.NumberColumn("Đo được TB (Wh)", format="%.1f"),
                    "pred": st.column_config.NumberColumn("Dự báo (Wh)", format="%.1f"),
                },
            )
            st.caption(f"Cột dự báo là của {model_name} tại chính các cấu hình đã bay đó.")
            ref = flights.merge(near[FEATURES], on=FEATURES)
        if seen:
            st.dataframe(
                ref[["flight", "date", "local_time", TARGET, "duration_s"]],
                hide_index=True,
                column_config={
                    "flight": st.column_config.NumberColumn("Chuyến #", format="%d"),
                    "date": "Ngày bay",
                    "local_time": "Giờ",
                    TARGET: st.column_config.NumberColumn("Năng lượng đo (Wh)", format="%.2f"),
                    "duration_s": st.column_config.NumberColumn("Thời lượng (s)", format="%.0f"),
                },
            )

    if has_profiles:
        st.subheader("Hồ sơ công suất của một chuyến bay thực")
        ref_ids = ref.flight.tolist()
        info = flights.set_index("flight")
        chosen = st.selectbox(
            "Chọn chuyến bay",
            ref_ids,
            format_func=lambda f: (
                f"Chuyến #{f} · {info.loc[f, 'payload']:.0f} g · {info.loc[f, 'speed']:.0f} m/s · "
                f"{info.loc[f, 'altitude']:.0f} m · {info.loc[f, TARGET]:.1f} Wh"
            ),
        )
        prof = get_csv("flight_profiles.csv")
        prof = prof[prof.flight == chosen]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38])
        fig.add_trace(go.Scatter(
            x=prof.time, y=prof.power_w, mode="lines", line=dict(color=MODEL_COLOR[LR], width=2),
            fill="tozeroy", fillcolor="rgba(42,120,214,0.12)",
            hovertemplate="t = %{x:.0f} s<br>Công suất %{y:.0f} W<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=prof.time, y=prof.rel_alt, mode="lines", line=dict(color=INK2, width=2),
            hovertemplate="t = %{x:.0f} s<br>Độ cao %{y:.0f} m<extra></extra>",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Công suất (W)", row=1, col=1)
        fig.update_yaxes(title_text="Độ cao tương đối (m)", row=2, col=1)
        fig.update_xaxes(title_text="Thời gian bay (s)", row=2, col=1)
        style(fig, height=420, legend=False)
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
        st.caption(
            f"Diện tích dưới đường công suất chính là năng lượng cả chuyến: chuyến #{chosen} tiêu thụ "
            f"{info.loc[chosen, TARGET]:.1f} Wh trong {info.loc[chosen, 'duration_s']:.0f} giây. "
            "Mô hình dự báo con số này trước khi drone cất cánh."
        )

    with st.expander("Con số này được tính thế nào?"):
        st.markdown(
            f"""
- **Đầu vào** chỉ gồm 3 tham số có trong kế hoạch bay: tải trọng, tốc độ hành trình và độ cao
  hành trình. Gió, vận tốc thực, thời lượng bay chỉ biết sau khi cất cánh nên không được dùng.
- **Đầu ra** là tổng năng lượng pin của cả chuyến, E (Wh) = (1/3600) · ∫ V(t) · I(t) dt, tính từ
  cảm biến dòng và áp của {summary['data']['n_flights']} chuyến bay thực.
- **Khoảng dự báo** = dự báo ± phân vị {int(level * 100)}% của sai số tuyệt đối ngoài mẫu. Với cấu
  hình đã có chuyến bay lặp trong dữ liệu, sai số lấy từ cross-validation; với cấu hình nội suy,
  sai số lấy từ phép thử "bỏ hẳn một cấu hình ra khỏi huấn luyện".
- **Kiểm tra độ phủ:** trên {summary['data']['n_holdout']} chuyến giữ kín, khoảng 90% của
  {model_name} chứa giá trị thực ở {summary['coverage_holdout'][model_name]['0.90'] * 100:.0f}% số chuyến.
- **Số chuyến mỗi pin** là ước tính tham khảo: (dung lượng danh định − dự trữ) chia cho cận trên
  của khoảng dự báo. Pin đã chai hoặc trời lạnh sẽ cho ít hơn.
"""
        )

# =============================== TAB 2: what-if ================================
with tab_whatif:
    st.subheader("Thay đổi từng tham số thì năng lượng thay đổi ra sao?")
    st.caption(
        f"Mỗi ô giữ nguyên hai tham số còn lại theo kế hoạch hiện tại "
        f"({payload} g · {speed:g} m/s · {altitude} m). Ba ô dùng chung trục dọc để so được mức ảnh hưởng."
    )
    mdl = models[model_name]
    color = MODEL_COLOR[model_name]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[FACTOR_LABEL[f] for f in FEATURES])
    for i, f in enumerate(FEATURES, start=1):
        grid = np.linspace(*RANGE[f], 81)
        frame = pd.DataFrame({k: np.full(81, float(plan[k].iloc[0])) for k in FEATURES})
        frame[f] = grid
        fig.add_trace(go.Scatter(
            x=grid, y=mdl.predict(frame), mode="lines", line=dict(color=color, width=2),
            name=f"Dự báo ({model_name})", legendgroup="pred", showlegend=i == 1,
            hovertemplate="%{x:.1f}<br>Dự báo %{y:.1f} Wh<extra></extra>",
        ), row=1, col=i)
        others = [k for k in FEATURES if k != f]
        match = flights[(flights[others[0]] == plan[others[0]].iloc[0]) & (flights[others[1]] == plan[others[1]].iloc[0])]
        if len(match):
            m = match.groupby(f)[TARGET].agg(["mean", "size"]).reset_index()
            fig.add_trace(go.Scatter(
                x=m[f], y=m["mean"], mode="markers", name="Đo thực (trung bình cấu hình)",
                legendgroup="actual", showlegend=False,
                marker=dict(size=10, symbol="diamond", color=INK2, line=dict(color=SURFACE, width=2)),
                customdata=m["size"],
                hovertemplate="%{x:g}<br>Đo được TB %{y:.1f} Wh (%{customdata} chuyến)<extra></extra>",
            ), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=[float(plan[f].iloc[0])], y=[pred], mode="markers", name="Kế hoạch hiện tại",
            legendgroup="plan", showlegend=i == 1,
            marker=dict(size=14, symbol="circle-open", color=INK, line=dict(color=INK, width=2.5)),
            hovertemplate="Kế hoạch hiện tại<br>%{y:.1f} Wh<extra></extra>",
        ), row=1, col=i)
    # legend entry for measured points even when the first panel has none
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Đo thực (trung bình cấu hình)", legendgroup="actual",
        marker=dict(size=10, symbol="diamond", color=INK2),
    ), row=1, col=1)
    fig.update_yaxes(title_text="Năng lượng cả chuyến (Wh)", row=1, col=1)
    style(fig, height=360)
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
    if not seen:
        st.caption("Điểm đo thực chỉ hiện khi hai tham số được giữ cố định trùng với một mức đã bay.")

    # best cruise speed for this payload and altitude
    sp = np.linspace(4, 12, 81)
    e_sp = mdl.predict(pd.DataFrame({"payload": payload, "speed": sp, "altitude": altitude}))
    best_sp, best_e = float(sp[e_sp.argmin()]), float(e_sp.min())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tốc độ ít tốn năng lượng nhất", f"{best_sp:g} m/s", border=True,
              help="Tìm trong khoảng 4–12 m/s, giữ nguyên tải trọng và độ cao của kế hoạch.")
    c2.metric("Năng lượng ở tốc độ đó", f"{best_e:.1f} Wh", border=True)
    c3.metric("Tiết kiệm so với kế hoạch", f"{pred - best_e:.1f} Wh", border=True,
              help=f"Tương đương {max(pred - best_e, 0) / pred * 100:.0f}% năng lượng dự báo của kế hoạch hiện tại.")

    st.subheader(f"Bản đồ năng lượng theo tốc độ và độ cao (tải {payload} g)")
    hm_speed, hm_alt = np.arange(4, 13, 1.0), np.array([25.0, 50.0, 75.0, 100.0])
    gs, ga = np.meshgrid(hm_speed, hm_alt)
    z = mdl.predict(pd.DataFrame({"payload": payload, "speed": gs.ravel(), "altitude": ga.ravel()})).reshape(gs.shape)
    fig = go.Figure(go.Heatmap(
        x=hm_speed, y=hm_alt, z=z, colorscale=[[i / 6, c] for i, c in enumerate(SEQ_BLUE)],
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Wh"), thickness=12, outlinewidth=0),
        hovertemplate="%{x:g} m/s · %{y:g} m<br>Dự báo %{z:.1f} Wh<extra></extra>",
    ))
    if speed in hm_speed and altitude in hm_alt:  # plan sits on a cell: ring around its label
        for width, col in [(7, SURFACE), (3, INK)]:
            fig.add_trace(go.Scatter(
                x=[speed], y=[altitude], mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(size=46, symbol="circle-open", color=col, line=dict(color=col, width=width)),
            ))
    else:  # plan falls between cells: small dot so it does not hide a label
        fig.add_trace(go.Scatter(
            x=[speed], y=[altitude], mode="markers", showlegend=False,
            marker=dict(size=14, color=INK, line=dict(color=SURFACE, width=3)),
            hovertemplate=f"Kế hoạch hiện tại<br>{speed:g} m/s · {altitude} m<br>{pred:.1f} Wh<extra></extra>",
        ))
    style(fig, height=330, legend=False)
    fig.update_xaxes(title_text="Tốc độ hành trình (m/s)", tickvals=hm_speed)
    fig.update_yaxes(title_text="Độ cao hành trình (m)", tickvals=hm_alt, showgrid=False)
    # cell labels in ink that contrasts with the cell: dark on light steps, white on dark steps
    z_lo, z_hi = float(z.min()), float(z.max())
    for r, a in enumerate(hm_alt):
        for c, s in enumerate(hm_speed):
            dark_cell = (z[r, c] - z_lo) / max(z_hi - z_lo, 1e-9) > 0.55
            fig.add_annotation(x=s, y=a, text=f"{z[r, c]:.1f}", showarrow=False,
                               font=dict(size=13, color="#ffffff" if dark_cell else INK))
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
    st.caption("Màu càng đậm càng tốn năng lượng. Dấu tròn đen đánh dấu kế hoạch hiện tại.")

    st.subheader("Mức liên hệ tuyến tính (RQ1)")
    coef = {c["term"]: c for c in summary["lr_coef"]}
    unit = {"payload": (100, "+100 g tải trọng"), "speed": (1, "+1 m/s tốc độ"), "altitude": (10, "+10 m độ cao")}
    coef_tbl = pd.DataFrame([
        {
            "Thay đổi": label,
            "Năng lượng thay đổi (Wh)": coef[f]["coef"] * k,
            "95% CI dưới": coef[f]["ci_lo"] * k,
            "95% CI trên": coef[f]["ci_hi"] * k,
        }
        for f, (k, label) in unit.items()
    ])
    st.dataframe(coef_tbl, hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="%+.2f") for c in coef_tbl.columns[1:]
    })
    st.caption(
        "Hệ số hồi quy tuyến tính (OLS) trên toàn bộ chuyến bay, giữ các tham số khác không đổi. "
        "Đây là mối liên hệ trong phạm vi dữ liệu, không phải kết luận nhân quả. Quan hệ với tốc độ "
        "thực tế là đường cong (xem ô giữa ở trên), nên hệ số tuyến tính chỉ là tóm tắt thô."
    )

# =============================== TAB 3: models =================================
with tab_models:
    d = summary["data"]
    st.subheader("Mô hình nào dự báo chính xác hơn? (RQ2)")
    st.caption(
        f"{d['n_holdout']} chuyến bay được giữ kín để kiểm tra cuối; {d['n_dev']} chuyến còn lại dùng cho "
        "cross-validation 5-fold lặp 10 lần. Đơn vị phân tích luôn là chuyến bay."
    )
    metric_tbl = pd.DataFrame([
        {
            "Mô hình": name,
            "MAE hold-out (Wh)": summary["holdout"][name]["MAE"],
            "RMSE hold-out (Wh)": summary["holdout"][name]["RMSE"],
            "R² hold-out": summary["holdout"][name]["R2"],
            "MAPE hold-out (%)": summary["holdout"][name]["MAPE"],
            "MAE CV (TB ± SD)": f"{summary['cv'][name]['MAE']['mean']:.2f} ± {summary['cv'][name]['MAE']['std']:.2f}",
            "RMSE CV (TB ± SD)": f"{summary['cv'][name]['RMSE']['mean']:.2f} ± {summary['cv'][name]['RMSE']['std']:.2f}",
        }
        for name in MODEL_NAMES
    ])
    st.dataframe(metric_tbl, hide_index=True, column_config={
        "MAE hold-out (Wh)": st.column_config.NumberColumn(format="%.2f"),
        "RMSE hold-out (Wh)": st.column_config.NumberColumn(format="%.2f"),
        "R² hold-out": st.column_config.NumberColumn(format="%.3f"),
        "MAPE hold-out (%)": st.column_config.NumberColumn(format="%.1f"),
    })

    left, right = st.columns([4, 6], gap="large")
    with left:
        st.markdown("**MAE trên 50 fold cross-validation**")
        fig = go.Figure()
        for name in MODEL_NAMES:
            g = folds[folds.model == name]
            fig.add_trace(go.Box(
                x=g.MAE, name=name, orientation="h", boxpoints="all", jitter=0.6, pointpos=0,
                marker=dict(color=MODEL_COLOR[name], size=5, opacity=0.55),
                line=dict(color=MODEL_COLOR[name], width=1.5), fillcolor="rgba(0,0,0,0)",
                hovertemplate="MAE %{x:.2f} Wh<extra>" + name + "</extra>",
            ))
        style(fig, height=360, legend=False, horizontal=True)
        fig.update_xaxes(title_text="MAE (Wh)", rangemode="tozero")
        fig.update_yaxes(categoryorder="array", categoryarray=MODEL_NAMES[::-1])
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
        st.caption("Mỗi chấm là một fold; hộp cho thấy trung vị và khoảng tứ phân vị.")
    with right:
        st.markdown(f"**Dự báo so với thực đo trên {d['n_holdout']} chuyến hold-out**")
        hold = per_flight[per_flight.split == "holdout"]
        lim = [math.floor(hold[TARGET].min() - 1), math.ceil(hold[TARGET].max() + 1)]
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=[f"{n}<br>MAE {summary['holdout'][n]['MAE']:.2f} Wh" for n in MODEL_NAMES])
        for i, name in enumerate(MODEL_NAMES, start=1):
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=AXIS, width=1),
                                     hoverinfo="skip"), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=hold[TARGET], y=hold[f"pred_{MODEL_KEYS[name]}"], mode="markers",
                marker=dict(size=8, color=MODEL_COLOR[name], line=dict(color=SURFACE, width=1.5)),
                customdata=hold[["flight", "config"]],
                hovertemplate="Chuyến #%{customdata[0]} (%{customdata[1]})<br>Thực đo %{x:.1f} Wh"
                              "<br>Dự báo %{y:.1f} Wh<extra></extra>",
            ), row=1, col=i)
            fig.update_xaxes(title_text="Thực đo (Wh)", range=lim, row=1, col=i)
        fig.update_yaxes(title_text="Dự báo (Wh)", range=lim, row=1, col=1)
        style(fig, height=360, legend=False)
        fig.update_layout(margin=dict(t=56))
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
        st.caption("Điểm càng sát đường chéo thì dự báo càng đúng.")

    st.subheader("Khác biệt có ý nghĩa thống kê không?")
    tests = summary["tests"]
    t = next(x for x in tests if x["model_a"] == LR and x["model_b"] == RF and x["metric"] == "MAE")
    better = RF if t["holdout_diff"] > 0 else LR
    excludes_zero = t["holdout_ci_lo"] > 0 or t["holdout_ci_hi"] < 0
    st.markdown(
        f"Trên {d['n_holdout']} chuyến hold-out, **{better}** có MAE thấp hơn "
        f"**{abs(t['holdout_diff']):.2f} Wh** (95% CI bootstrap theo chuyến bay: "
        f"{min(abs(t['holdout_ci_lo']), abs(t['holdout_ci_hi'])):.2f} đến "
        f"{max(abs(t['holdout_ci_lo']), abs(t['holdout_ci_hi'])):.2f} Wh). "
        f"Corrected resampled t-test trên 50 fold cho {fmt_p(t['cv_p'], sentence=True)}. "
        + ("Khoảng tin cậy không chứa 0, nên khác biệt giữa hai mô hình là có ý nghĩa thống kê."
           if excludes_zero else
           "Khoảng tin cậy chứa 0, nên chưa đủ bằng chứng để nói hai mô hình khác nhau.")
    )
    test_tbl = pd.DataFrame([
        {
            "So sánh": f"{x['model_a']} − {x['model_b']}",
            "Chỉ số": x["metric"],
            "Chênh lệch CV (Wh)": x["cv_mean_diff"],
            "p (corrected t-test)": fmt_p(x["cv_p"]),
            "Chênh lệch hold-out (Wh)": x["holdout_diff"],
            "95% CI dưới": x["holdout_ci_lo"],
            "95% CI trên": x["holdout_ci_hi"],
        }
        for x in tests
    ])
    st.dataframe(test_tbl, hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="%+.2f")
        for c in ["Chênh lệch CV (Wh)", "Chênh lệch hold-out (Wh)", "95% CI dưới", "95% CI trên"]
    })
    st.caption(
        "Chênh lệch = sai số của mô hình thứ nhất trừ mô hình thứ hai; số dương nghĩa là mô hình thứ hai "
        "chính xác hơn. Corrected resampled t-test (Nadeau & Bengio, 2003) hiệu chỉnh cho việc các fold "
        "dùng chung dữ liệu huấn luyện; CI trên hold-out lấy từ 10,000 lần bootstrap theo chuyến bay."
    )

# =============================== TAB 4: errors =================================
with tab_errors:
    st.subheader("Mô hình sai nhiều ở đâu? (RQ3)")
    st.caption(
        "Mỗi chuyến bay có một dự báo ngoài mẫu: các chuyến dùng để huấn luyện lấy từ cross-validation, "
        "các chuyến giữ kín lấy từ lần kiểm tra cuối."
    )
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[FACTOR_LABEL[f] for f in FEATURES])
    for i, f in enumerate(FEATURES, start=1):
        for name in MODEL_NAMES:
            g = per_flight.groupby(f)[f"abs_err_{MODEL_KEYS[name]}"].agg(["mean", "size"]).reset_index()
            fig.add_trace(go.Scatter(
                x=g[f], y=g["mean"], mode="lines+markers", name=name, legendgroup=name, showlegend=i == 1,
                line=dict(color=MODEL_COLOR[name], width=2),
                marker=dict(size=9, color=MODEL_COLOR[name], line=dict(color=SURFACE, width=2)),
                customdata=g["size"],
                hovertemplate=name + "<br>Mức %{x:g}<br>MAE %{y:.2f} Wh (%{customdata} chuyến)<extra></extra>",
            ), row=1, col=i)
        fig.update_xaxes(tickvals=sorted(per_flight[f].unique()), row=1, col=i)
    fig.update_yaxes(title_text="MAE (Wh)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero")
    style(fig, height=360)
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)

    with st.expander("Xem dạng bảng"):
        level_tbl = pd.concat([
            per_flight.groupby(f)[[f"abs_err_{MODEL_KEYS[n]}" for n in MODEL_NAMES]].mean()
            .set_axis(MODEL_NAMES, axis=1).rename_axis("Mức").reset_index()
            .assign(**{"Tham số": FACTOR_LABEL[f]})
            for f in FEATURES
        ])[["Tham số", "Mức", *MODEL_NAMES]]
        st.dataframe(level_tbl, hide_index=True, column_config={
            n: st.column_config.NumberColumn(f"MAE {n} (Wh)", format="%.2f") for n in MODEL_NAMES
        })

    left, right = st.columns([9, 11], gap="large")
    with left:
        st.markdown("**Khi gặp cấu hình chưa từng bay**")
        hold_mae = [summary["holdout"][n]["MAE"] for n in MODEL_NAMES]
        loco_mae = [summary["loco"][n]["MAE"] for n in MODEL_NAMES]
        fig = go.Figure()
        for name, a, b in zip(MODEL_NAMES, hold_mae, loco_mae):
            fig.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines",
                                     line=dict(color=AXIS, width=2), showlegend=False, hoverinfo="skip"))
        # value labels sit outside each pair so two close values never collide
        for label, vals, other, col in [
            ("Cấu hình đã có chuyến lặp (hold-out)", hold_mae, loco_mae, "#86b6ef"),
            ("Cấu hình bị bỏ hẳn khỏi huấn luyện", loco_mae, hold_mae, "#1c5cab"),
        ]:
            fig.add_trace(go.Scatter(
                x=vals, y=MODEL_NAMES, mode="markers+text", name=label,
                marker=dict(size=13, color=col, line=dict(color=SURFACE, width=2)),
                text=[f"{v:.2f}" for v in vals],
                textposition=["middle left" if v <= o else "middle right" for v, o in zip(vals, other)],
                textfont=dict(color=INK, size=12),
                hovertemplate="%{y}<br>MAE %{x:.2f} Wh<extra>" + label + "</extra>",
            ))
        style(fig, height=300, horizontal=True)
        fig.update_xaxes(title_text="MAE (Wh)", range=[0, max(hold_mae + loco_mae) * 1.2])
        fig.update_yaxes(categoryorder="array", categoryarray=MODEL_NAMES[::-1])
        fig.update_layout(legend=dict(orientation="v", y=1.02, yanchor="bottom"), margin=dict(t=70))
        st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
        rf_h, rf_l = summary["holdout"][RF]["MAE"], summary["loco"][RF]["MAE"]
        po_h, po_l = summary["holdout"][POLY]["MAE"], summary["loco"][POLY]["MAE"]
        st.caption(
            f"Random Forest sai nhiều hơn hẳn khi phải dự báo một cấu hình chưa từng thấy "
            f"({rf_h:.2f} → {rf_l:.2f} Wh), vì cây quyết định không nội suy được giữa các mức. "
            f"Polynomial Ridge thay đổi ít hơn ({po_h:.2f} → {po_l:.2f} Wh)."
        )
    with right:
        st.markdown(f"**10 chuyến bay {model_name} dự báo sai nhất**")
        k = MODEL_KEYS[model_name]
        worst = per_flight.nlargest(10, f"abs_err_{k}")
        worst = worst.assign(
            cfg=[config_label(p, s, a) for p, s, a in worst[FEATURES].itertuples(index=False)]
        )[["flight", "cfg", TARGET, f"pred_{k}", f"abs_err_{k}"]]
        st.dataframe(worst, hide_index=True, column_config={
            "flight": st.column_config.NumberColumn("Chuyến #", format="%d"),
            "cfg": "Cấu hình",
            TARGET: st.column_config.NumberColumn("Thực đo (Wh)", format="%.1f"),
            f"pred_{k}": st.column_config.NumberColumn("Dự báo (Wh)", format="%.1f"),
            f"abs_err_{k}": st.column_config.NumberColumn("Sai số (Wh)", format="%.2f"),
        })

# =============================== TAB 5: data ===================================
with tab_data:
    d = summary["data"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chuyến bay dùng để mô hình hoá", f"{d['n_flights']}", border=True)
    k2.metric("Cấu hình tải × tốc độ × độ cao", f"{d['n_configs']}", border=True)
    k3.metric("Năng lượng mỗi chuyến", f"{d['energy_min']:.0f} – {d['energy_max']:.0f} Wh", border=True)
    k4.metric("Năng lượng trung bình", f"{d['energy_mean']:.1f} Wh", border=True)

    st.subheader("Toàn bộ dữ liệu trong một hình")
    alts = sorted(flights.altitude.unique())
    fig = make_subplots(rows=1, cols=len(alts), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"Độ cao {a:.0f} m" for a in alts])
    for i, a in enumerate(alts, start=1):
        for p, col in PAYLOAD_COLOR.items():
            g = config_stats[(config_stats.altitude == a) & (config_stats.payload == p)].sort_values("speed")
            fig.add_trace(go.Scatter(
                x=g.speed, y=g["mean"], mode="lines+markers", name=f"Tải {p} g", legendgroup=str(p),
                showlegend=i == 1, line=dict(color=col, width=2),
                marker=dict(size=9, color=col, line=dict(color=SURFACE, width=2)),
                customdata=g[["n", "lo", "hi"]],
                hovertemplate=f"Tải {p} g · %{{x:g}} m/s · {a:.0f} m<br>TB %{{y:.1f}} Wh "
                              "(%{customdata[0]} chuyến, %{customdata[1]:.1f}–%{customdata[2]:.1f})<extra></extra>",
            ), row=1, col=i)
        fig.update_xaxes(title_text="Tốc độ (m/s)", tickvals=[4, 6, 8, 10, 12], row=1, col=i)
    fig.update_yaxes(title_text="Năng lượng TB mỗi chuyến (Wh)", row=1, col=1)
    style(fig, height=360)
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)
    st.caption(
        "Mỗi điểm là trung bình các chuyến bay lặp của một cấu hình. Bay cao hơn và chở nặng hơn thì tốn "
        "hơn; bay chậm tốn hơn vì drone phải ở trên không lâu hơn."
    )

    with st.expander("Số chuyến bay của từng cấu hình"):
        counts = config_stats.pivot_table(index=["payload", "altitude"], columns="speed", values="n", fill_value=0)
        counts.columns = [f"{c:g} m/s" for c in counts.columns]
        counts = counts.astype(int).reset_index().rename(columns={"payload": "Tải (g)", "altitude": "Độ cao (m)"})
        st.dataframe(counts, hide_index=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Nguồn dữ liệu")
        st.markdown(
            """
- Rodrigues, T. A. et al. (2021). *Data Collected with Package Delivery Quadcopter Drone.*
  Carnegie Mellon University, KiltHub. DOI 10.1184/R1/12683453 (CC BY 4.0).
- 209 bản ghi bay của DJI Matrice 100, cảm biến dòng và áp đồng bộ khoảng 5 Hz.
- Tập mô hình hoá: 181 chuyến tuyến R1 (3 mức tải × 5 mức tốc độ × 4 mức độ cao), mỗi chuyến
  gộp thành một dòng dữ liệu.
"""
        )
    with right:
        st.subheader("Giới hạn cần nói rõ")
        st.markdown(
            """
- Chỉ đúng cho một mẫu drone (Matrice 100), một tuyến bay (R1), một khu vực bay.
- Chỉ trong phạm vi tải 0–500 g, tốc độ 4–12 m/s, độ cao 25–100 m; ứng dụng không cho nhập ngoài
  phạm vi này.
- Chưa tính gió: bộ dữ liệu không có nguồn gió biết trước khi bay.
- Là công cụ tham khảo khi lập kế hoạch; dùng để ra quyết định vận hành thật cần kiểm chứng thêm.
"""
        )
