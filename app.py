"""EDA descriptivo para tablas de miles a millones de filas.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.ndimage import gaussian_filter

import eda_core as ec

st.set_page_config(page_title="EDA Big Data", page_icon="📊", layout="wide")

SECTIONS = ["📁 Datos y perfil", "🧹 Calidad", "📈 Univariado", "🔀 Bivariado",
            "🧮 Correlaciones", "🕒 Temporal", "🤖 AI Ready", "💾 Exportación"]
FREQS = {"hora": "1h", "día": "1d", "semana": "1w", "mes": "1mo", "trimestre": "1q", "año": "1y"}
PCT = st.column_config.NumberColumn(format="percent")


# --------------------------------------------------------------------------- #
# Caché
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Leyendo archivo…", max_entries=2)
def load_upload(key: str, _data: bytes, name: str, sep: str, max_rows: int | None) -> pl.DataFrame:
    return ec.normalize(ec.read_bytes(_data, name, sep, max_rows))


@st.cache_resource(show_spinner="Leyendo desde disco…", max_entries=2)
def load_path(path: str, mtime: float, sep: str, max_rows: int | None) -> pl.DataFrame:
    return ec.normalize(ec.read_path(path, sep, max_rows))


@st.cache_resource(show_spinner="Generando datos sintéticos…", max_entries=2)
def load_synthetic(n: int, seed: int) -> pl.DataFrame:
    return ec.synthetic(n, seed)


@st.cache_resource(show_spinner="Muestreando…", max_entries=2)
def global_sample(key: str, _df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return ec.maybe_sample(_df, n, seed)


@st.cache_data(show_spinner="Calculando…", max_entries=128)
def cached(fn: str, key: str, _df: pl.DataFrame, *args, **kwargs):
    """Memoiza cualquier función de eda_core. `key` identifica el dataset en uso."""
    return getattr(ec, fn)(_df, *args, **kwargs)


# --------------------------------------------------------------------------- #
# Utilidades de presentación
# --------------------------------------------------------------------------- #
def fp(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/d"
    return "< 1e-300" if p == 0 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def f4(v) -> str:
    return "n/d" if v is None or not np.isfinite(v) else f"{v:.4f}"


def show(fig, h: int | None = None):
    if h:
        fig.update_layout(height=h)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")


def big_n_note():
    st.caption("ℹ️ Con cientos de miles de filas casi cualquier diferencia es 'significativa' (p≈0). "
               "Decide por el **tamaño del efecto** (r, η², ε², V), no por el p-valor.")


def need(cols: list, n: int, what: str) -> bool:
    if len(cols) < n:
        st.info(f"Se necesitan al menos {n} columnas {what}.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Barra lateral: datos y parámetros
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 EDA Big Data")
source = st.sidebar.radio("Fuente de datos", ["Datos sintéticos", "Subir archivo", "Ruta local"])
df_base, ds_name, src_key = None, "", ""

with st.sidebar.expander("Opciones de lectura", expanded=source != "Datos sintéticos"):
    sep = st.selectbox("Separador CSV", [",", ";", "|", "\t"], format_func=lambda s: {"\t": "tab"}.get(s, s))
    max_rows = st.number_input("Leer como máximo N filas (0 = todas)", 0, step=100_000, value=0) or None

try:
    if source == "Datos sintéticos":
        n_syn = st.sidebar.select_slider("Filas", [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000], 1_000_000,
                                         format_func=lambda v: f"{v:,}")
        df_base, ds_name, src_key = load_synthetic(n_syn, 42), "sintetico_clientes", f"syn{n_syn}"
    elif source == "Subir archivo":
        up = st.sidebar.file_uploader("CSV, TSV, Parquet o JSONL", type=["csv", "tsv", "txt", "parquet", "jsonl", "ndjson"])
        if up is not None:
            src_key = f"up|{up.name}|{up.size}|{getattr(up, 'file_id', '')}|{sep}|{max_rows}"
            df_base, ds_name = load_upload(src_key, up.getvalue(), up.name, sep, max_rows), up.name.rsplit(".", 1)[0]
    else:
        path = st.sidebar.text_input("Ruta (admite globs: datos/*.parquet)", placeholder=r"C:\datos\ventas.parquet")
        if path:
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            df_base = load_path(path, mtime, sep, max_rows)
            ds_name, src_key = os.path.basename(path).rsplit(".", 1)[0], f"path|{path}|{mtime}|{sep}|{max_rows}"
except Exception as e:  # noqa: BLE001 - mostrar cualquier error de lectura al usuario
    st.error(f"No se pudo leer los datos: {e}")
    st.stop()

if df_base is None:
    st.title("📊 EDA para Big Data")
    st.info("Elige una fuente de datos en la barra lateral para empezar.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Escala")
seed = int(st.sidebar.number_input("Semilla", 0, value=42))
g_n = int(st.sidebar.number_input(
    "Muestra global (0 = todas las filas)", 0, step=100_000, value=0,
    help="Para tablas enormes: todo el análisis trabaja sobre esta muestra aleatoria."))
plot_n = int(st.sidebar.number_input("Puntos máx. en gráficos (scatter/hexbin/violín)", 1_000,
                                     step=10_000, value=50_000))
test_n = int(st.sidebar.number_input("Filas máx. para métricas costosas (Kendall, Kruskal, matrices)",
                                     1_000, step=10_000, value=100_000))
top_k = int(st.sidebar.slider("Top-K categorías (resto → 'Otros')", 3, 50, 15))

df = global_sample(src_key, df_base, g_n, seed) if g_n else df_base
low_card = cached("low_card_int_cols", f"{src_key}|{g_n}|{seed}", df)
as_cat = st.sidebar.multiselect("Enteros a tratar como categóricos", [c for c, t in df.schema.items() if t.is_integer()],
                                default=low_card, help="Por defecto: enteros con ≤ 10 valores distintos.")

KEY = f"{src_key}|{g_n}|{seed}|{sorted(as_cat)}"
kinds = ec.column_kinds(df, as_cat)
NUMS = [c for c, k in kinds.items() if k == ec.NUM]
CATS = [c for c, k in kinds.items() if k == ec.CAT]
DATES = [c for c, k in kinds.items() if k == ec.DATE]

st.sidebar.divider()
section = st.sidebar.radio("Sección", SECTIONS)
st.sidebar.caption(f"**{ds_name}** · {df.height:,} filas × {df.width} columnas"
                   + (f" (muestra de {df_base.height:,})" if g_n and df_base.height > g_n else ""))


def get_profile():
    return cached("profile", KEY, df, kinds)


# Columnas con pinta de ID / constantes van al final de los selectores: rara vez interesa analizarlas.
_skip = set(get_profile().filter(pl.col("alertas").str.contains("posible ID|constante|vacía"))["columna"])
NUMS = [c for c in NUMS if c not in _skip] + [c for c in NUMS if c in _skip]
CATS = [c for c in CATS if c not in _skip] + [c for c in CATS if c in _skip]


def get_nsum():
    return cached("numeric_summary", KEY, df, NUMS)


def get_dups():
    return cached("duplicate_rows", KEY, df)


# --------------------------------------------------------------------------- #
# 1. Datos y perfil
# --------------------------------------------------------------------------- #
def page_profile():
    st.header("📁 Datos y perfil")
    ov = cached("overview", KEY, df, kinds)
    c = st.columns(6)
    c[0].metric("Filas", f"{ov['filas'] / 1e6:,.2f} M" if ov["filas"] >= 1e6 else f"{ov['filas']:,}",
                help=f"{ov['filas']:,}")
    c[1].metric("Columnas", ov["columnas"])
    c[2].metric("Memoria", f"{ov['memoria_mb']:,.1f} MB")
    c[3].metric("Celdas nulas", f"{ov['pct_celdas_nulas']:.2%}")
    c[4].metric("Num / Cat / Fecha", f"{ov[ec.NUM]} / {ov[ec.CAT]} / {ov[ec.DATE]}")
    c[5].metric("Filas duplicadas", f"{get_dups():,}")

    st.subheader("Perfil de columnas")
    prof = get_profile()
    st.dataframe(prof, width="stretch", hide_index=True, column_config={
        "pct_nulos": st.column_config.ProgressColumn("% nulos", format="percent", min_value=0, max_value=1),
        "pct_unicos": st.column_config.NumberColumn("% únicos", format="percent"),
        "pct_top": st.column_config.NumberColumn("% top", format="percent"),
        "n_unicos": st.column_config.NumberColumn("únicos (aprox)", format="localized"),
        "top_valor": "valor más frecuente"})
    st.caption("Cardinalidad aproximada (HyperLogLog) para escalar a millones de filas.")

    st.subheader("Vista previa")
    mode = st.radio("Filas", ["Primeras", "Aleatorias"], horizontal=True, label_visibility="collapsed")
    st.dataframe(df.head(100) if mode == "Primeras" else df.sample(min(100, df.height), seed=seed),
                 width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 2. Calidad
# --------------------------------------------------------------------------- #
def page_quality():
    st.header("🧹 Calidad de datos")
    prof = get_profile()
    alerts = prof.filter(pl.col("alertas") != "").select("columna", "clase", "pct_nulos", "n_unicos", "pct_top", "alertas")
    st.subheader(f"Alertas ({alerts.height})")
    if alerts.height:
        st.dataframe(alerts, hide_index=True, width="stretch",
                     column_config={"pct_nulos": PCT, "pct_top": PCT})
    else:
        st.success("Sin alertas en el perfil de columnas.")

    dups = get_dups()
    if dups > 0:
        st.warning(f"**{dups:,}** filas duplicadas exactas ({dups / df.height:.2%}).")

    nulls = prof.filter(pl.col("nulos") > 0).sort("pct_nulos", descending=True)
    l, r = st.columns(2)
    with l:
        st.subheader("Nulos por columna")
        if nulls.height:
            fig = px.bar(nulls.to_pandas(), x="pct_nulos", y="columna", orientation="h", text_auto=".1%")
            fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            show(fig, 40 + 28 * nulls.height)
        else:
            st.success("Ninguna columna tiene nulos.")
    with r:
        st.subheader("Patrones de nulos")
        pat = cached("null_patterns", KEY, df)
        if pat is not None:
            st.dataframe(pat, hide_index=True, width="stretch", column_config={"pct": PCT})
            st.caption("Combinaciones de columnas que están nulas a la vez en la misma fila.")

    nc = cached("nullity_corr", KEY, df, test_n, seed)
    if nc is not None:
        st.subheader("Correlación de nulidad")
        st.caption("¿Faltan juntas? Valores altos sugieren un mismo origen del faltante (no aleatorio).")
        show(px.imshow(nc, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f"),
             120 + 30 * len(nc))


# --------------------------------------------------------------------------- #
# 3. Univariado
# --------------------------------------------------------------------------- #
def page_univariate():
    st.header("📈 Univariado")
    kind = st.radio("Tipo", ["Numéricas", "Categóricas", "Fechas"], horizontal=True)

    if kind == "Numéricas":
        if not need(NUMS, 1, "numéricas"):
            return
        nsum = get_nsum()
        with st.expander("Resumen de todas las numéricas", expanded=True):
            st.dataframe(nsum, hide_index=True, width="stretch", column_config={
                "pct_ceros": PCT, "pct_negativos": PCT, "pct_outliers": PCT})
            st.caption("Estadísticos sobre todas las filas. Outliers = fuera de [p25 − 1.5·IQR, p75 + 1.5·IQR].")
        col = st.selectbox("Variable", NUMS)
        c1, c2, c3 = st.columns(3)
        bins = c1.slider("Bins", 10, 200, 60)
        clip = c2.toggle("Recortar p0.5–p99.5", True)
        logx = c3.toggle("Escala log1p", False)
        h = cached("histogram", KEY, df, col, bins, clip, logx)
        row = nsum.filter(pl.col("variable") == col).to_dicts()[0]
        m = st.columns(6)
        for i, (lab, k) in enumerate([("Media", "media"), ("Mediana", "mediana"), ("Desv.", "desv"),
                                      ("Asimetría", "asimetria"), ("Curtosis", "curtosis"), ("% outliers", "pct_outliers")]):
            v = row[k]
            m[i].metric(lab, "n/d" if v is None else (f"{v:.2%}" if k == "pct_outliers" else f"{v:,.4g}"))
        if h:
            e = h["edges"]
            fig = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h["counts"], width=np.diff(e), marker_line_width=0))
            fig.update_layout(title=f"Histograma de {col}" + (" (log1p)" if logx else ""), bargap=0,
                              xaxis_title=col, yaxis_title="frecuencia")
            show(fig, 380)
            if h["fuera_rango"]:
                st.caption(f"{h['fuera_rango']:,} valores fuera del rango mostrado (recorte de colas).")
        q = row
        box = go.Figure(go.Box(name=col, q1=[q["p25"]], median=[q["mediana"]], q3=[q["p75"]], mean=[q["media"]],
                               lowerfence=[max(q["min"], q["p25"] - 1.5 * (q["p75"] - q["p25"]))],
                               upperfence=[min(q["max"], q["p75"] + 1.5 * (q["p75"] - q["p25"]))],
                               orientation="h", boxmean=True))
        box.update_layout(title="Boxplot (calculado con cuantiles de todas las filas)")
        show(box, 200)

    elif kind == "Categóricas":
        if not need(CATS, 1, "categóricas"):
            return
        col = st.selectbox("Variable", CATS)
        k = st.slider("Mostrar top", 5, 100, min(top_k, 30))
        tv, nlev = cached("top_values", KEY, df, col, k)
        st.metric("Categorías distintas", f"{nlev:,}")
        pdf = tv.to_pandas()
        fig = go.Figure()
        fig.add_bar(x=pdf["valor"], y=pdf["n"], name="frecuencia")
        fig.add_scatter(x=pdf["valor"], y=pdf["pct_acum"], name="% acumulado", yaxis="y2", mode="lines+markers")
        fig.update_layout(title=f"Pareto de {col}", yaxis2=dict(overlaying="y", side="right", tickformat=".0%",
                                                                range=[0, 1.05]), legend=dict(orientation="h"))
        show(fig, 420)
        st.dataframe(tv, hide_index=True, width="stretch", column_config={"pct": PCT, "pct_acum": PCT})

    else:
        if not need(DATES, 1, "de fecha"):
            return
        col = st.selectbox("Variable", DATES)
        freq = st.select_slider("Periodo", list(FREQS), "mes")
        mn, mx = df.select(pl.col(col).min().alias("a"), pl.col(col).max().alias("b")).row(0)
        st.write(f"Rango: **{mn}** → **{mx}**")
        ts = cached("time_series", KEY, df, col, FREQS[freq])
        show(px.line(ts.to_pandas(), x="periodo", y="valor", title=f"Filas por {freq}"), 380)


# --------------------------------------------------------------------------- #
# 4. Bivariado
# --------------------------------------------------------------------------- #
def num_num():
    if not need(NUMS, 2, "numéricas"):
        return
    c1, c2 = st.columns(2)
    x = c1.selectbox("X", NUMS, 0)
    y = c2.selectbox("Y", [c for c in NUMS if c != x], 0)
    r = cached("pair_correlations", KEY, df, x, y, test_n, seed)
    if r["n"] < 3:
        st.warning("Muy pocas filas completas para este par.")
        return
    m = st.columns(4)
    lo, hi = r["pearson_ic95"]
    m[0].metric(f"Pearson r · {ec.strength(r['pearson'])}", f4(r["pearson"]),
                help=f"Lineal, todas las filas (n={r['n']:,}). IC95% [{lo:.4f}, {hi:.4f}], p={fp(r['p_pearson'])}")
    m[1].metric(f"Spearman ρ · {ec.strength(r['spearman'])}", f4(r["spearman"]),
                help=f"Monótona por rangos, todas las filas. p={fp(r['p_spearman'])}")
    m[2].metric(f"Kendall τ · {ec.strength(r['kendall'])}", f4(r["kendall"]),
                help=f"Concordancia de pares, muestra n={r['n_kendall']:,}. p={fp(r['p_kendall'])}")
    m[3].metric("Pendiente OLS", f"{r['pendiente']:.4g}", help=f"y ≈ {r['intercepto']:.4g} + {r['pendiente']:.4g}·x")
    if abs((r["spearman"] or 0) - (r["pearson"] or 0)) > 0.1:
        st.caption("⚠️ Pearson y Spearman difieren > 0.1: relación no lineal o efecto de outliers.")
    big_n_note()

    o1, o2, o3 = st.columns(3)
    bins = o1.slider("Resolución (bins)", 20, 200, 80)
    clip = o2.toggle("Recortar colas p0.5–p99.5", True)
    logc = o3.toggle("Color en escala log", True)
    tabs = st.tabs(["Hexbin", "Heatmap 2D", "Contour", "Scatter (muestra)", "Tendencia por bins"])
    h = cached("hist2d", KEY, df, x, y, bins, clip)
    z = np.log1p(h["H"].T) if logc else h["H"].T
    with tabs[0]:
        xa, ya = cached("xy_arrays", KEY, df, x, y, max(plot_n, 1_000_000), seed)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        hb = ax.hexbin(xa, ya, gridsize=bins // 2, bins="log" if logc else None, cmap="viridis", mincnt=1,
                       extent=(*h["xr"], *h["yr"]))
        fig.colorbar(hb, ax=ax, label="log10(n)" if logc else "n")
        ax.set(xlabel=x, ylabel=y)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Hexbin sobre {len(xa):,} puntos (agrega en celdas hexagonales; escala a millones).")
    with tabs[1]:
        fig = go.Figure(go.Heatmap(x=h["xc"], y=h["yc"], z=z, colorscale="Viridis",
                                   colorbar_title="log(1+n)" if logc else "n"))
        fig.update_layout(xaxis_title=x, yaxis_title=y)
        show(fig, 500)
        st.caption(f"Histograma 2D con todas las filas completas ({h['n']:,}); {h['en_rango']:,} dentro del rango.")
    with tabs[2]:
        fig = go.Figure(go.Contour(x=h["xc"], y=h["yc"], z=gaussian_filter(z, 1.2), colorscale="Viridis",
                                   contours_coloring="heatmap", line_width=0.5, ncontours=15))
        fig.update_layout(xaxis_title=x, yaxis_title=y)
        show(fig, 500)
        st.caption("Contornos de densidad (histograma 2D suavizado). Útil para ver modos y subpoblaciones.")
    with tabs[3]:
        color = st.selectbox("Color por", ["(ninguno)"] + CATS, key="sc_color")
        n_sc = min(plot_n, 200_000)
        s = cached("scatter_sample", KEY, df, x, y, n_sc, seed, None if color == "(ninguno)" else color, 10)
        fig = px.scatter(s, x="x", y="y", color="color" if "color" in s else None, opacity=0.35, render_mode="webgl",
                         labels={"x": x, "y": y})
        fig.update_traces(marker_size=3)
        xs = np.array([s["x"].min(), s["x"].max()])
        fig.add_scatter(x=xs, y=r["intercepto"] + r["pendiente"] * xs, mode="lines", name="OLS (todas las filas)",
                        line=dict(color="red", dash="dash"))
        show(fig, 500)
        st.caption(f"Muestra aleatoria de {len(s):,} puntos.")
    with tabs[4]:
        q = st.slider("Cuantiles de X", 5, 100, 20)
        bt = cached("binned_trend", KEY, df, x, y, q).to_pandas()
        fig = go.Figure([
            go.Scatter(x=bt["x"], y=bt["y_p90"], line_width=0, showlegend=False),
            go.Scatter(x=bt["x"], y=bt["y_p10"], fill="tonexty", line_width=0, name="p10–p90",
                       fillcolor="rgba(99,110,250,0.2)"),
            go.Scatter(x=bt["x"], y=bt["y_media"], mode="lines+markers", name="media"),
            go.Scatter(x=bt["x"], y=bt["y_mediana"], mode="lines", name="mediana", line_dash="dot")])
        fig.update_layout(xaxis_title=f"{x} (media del bin)", yaxis_title=y)
        show(fig, 450)
        st.caption("Media/mediana de Y por cuantil de X con todas las filas: muestra la forma de la relación "
                   "(no linealidad, saturación) sin dibujar millones de puntos.")


def num_cat():
    if not (need(NUMS, 1, "numéricas") and need(CATS, 1, "categóricas")):
        return
    c1, c2, c3 = st.columns(3)
    num = c1.selectbox("Numérica", NUMS)
    cat = c2.selectbox("Categórica", CATS)
    order = c3.selectbox("Ordenar grupos por", ["n", "media", "mediana", "p90"])
    g = cached("group_stats", KEY, df, num, cat, top_k).sort(order, descending=True)
    an = ec.anova_from_groups(g)
    per_group = max(200, test_n // max(g.height, 1))
    kw = cached("kruskal_test", KEY, df, num, cat, top_k, per_group, seed)
    m = st.columns(4)
    if an:
        m[0].metric("ANOVA F", f"{an['F']:,.2f}", help=f"gl={an['gl']}, p={fp(an['p'])}. Exacto con todas las filas.")
        m[1].metric("η² (ANOVA)", f4(an["eta2"]), help="Proporción de varianza explicada por el grupo. "
                    "~0.01 pequeño, ~0.06 mediano, ≥0.14 grande.")
    if kw:
        m[2].metric("Kruskal-Wallis H", f"{kw['H']:,.2f}",
                    help=f"No paramétrico, muestra estratificada n={kw['n']:,}. p={fp(kw['p'])}")
        m[3].metric("ε² (Kruskal)", f4(kw["epsilon2"]))
    big_n_note()
    st.caption(f"Se muestran las {top_k} categorías más frecuentes; el resto se agrupa en '{ec.OTHER_LABEL}'.")

    tabs = st.tabs(["Mean / Median / P90", "Boxplot", "Violin (muestra)", "Tabla"])
    gp = g.to_pandas()
    with tabs[0]:
        stats_sel = st.multiselect("Estadísticos", ["media", "mediana", "p90", "p10"], ["media", "mediana", "p90"])
        if stats_sel:
            long = gp.melt(id_vars="grupo", value_vars=stats_sel, var_name="estadístico", value_name=num)
            fig = px.bar(long, x="grupo", y=num, color="estadístico", barmode="group")
            fig.update_layout(xaxis_title=cat)
            show(fig, 420)
    with tabs[1]:
        fig = go.Figure(go.Box(x=gp["grupo"], q1=gp["p25"], median=gp["mediana"], q3=gp["p75"],
                               lowerfence=gp["lim_inf"], upperfence=gp["lim_sup"], mean=gp["media"],
                               boxmean=True, name=num))
        fig.update_layout(xaxis_title=cat, yaxis_title=num)
        show(fig, 450)
        st.caption("Cajas calculadas con cuantiles de todas las filas (no se dibujan los outliers individuales).")
    with tabs[2]:
        s = cached("stratified_sample", KEY, df, num, cat, top_k, max(100, plot_n // max(g.height, 1)), seed)
        clip_v = st.toggle("Recortar p1–p99 (visual)", True, key="vclip")
        sp = s.to_pandas()
        if clip_v and len(sp):
            lo, hi = sp["y"].quantile([.01, .99])
            sp = sp[(sp["y"] >= lo) & (sp["y"] <= hi)]
        fig = px.violin(sp, x="grupo", y="y", box=True, points=False,
                        category_orders={"grupo": gp["grupo"].tolist()}, labels={"y": num, "grupo": cat})
        show(fig, 450)
        st.caption(f"Muestra estratificada: hasta {max(100, plot_n // max(g.height, 1)):,} filas por grupo.")
    with tabs[3]:
        st.dataframe(g, hide_index=True, width="stretch")


def cat_cat():
    if not need(CATS, 2, "categóricas"):
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox("Filas (A)", CATS, 0)
    b = c2.selectbox("Columnas (B)", [c for c in CATS if c != a], 0)
    ct = cached("crosstab", KEY, df, a, b, top_k)
    t = ct["table"]
    res = ec.cat_association(t)
    m = st.columns(4)
    m[0].metric(f"V de Cramér · {ec.strength(res['cramers_v'])}", f4(res["cramers_v"]), help="Con corrección de sesgo (Bergsma). 0 = independencia, 1 = asociación perfecta.")
    m[1].metric("χ²", f"{res['chi2']:,.1f}", help=f"gl={res['gl']}, p={fp(res['p'])}")
    m[2].metric(f"U de Theil ({a}|{b})", f4(res["theil_u_ab"]), help=f"Cuánto reduce {b} la incertidumbre de {a} (asimétrico).")
    m[3].metric(f"U de Theil ({b}|{a})", f4(res["theil_u_ba"]), help=f"Cuánto reduce {a} la incertidumbre de {b}.")
    big_n_note()

    tabs = st.tabs(["Heatmap", "Barras apiladas 100%"])
    with tabs[0]:
        mode = st.radio("Valor", ["Conteo", "% fila", "% columna", "Residuos estandarizados"], horizontal=True)
        if mode == "Conteo":
            z, fmt, scale = t, ",.0f", "Viridis"
        elif mode == "% fila":
            z, fmt, scale = t / t.sum(1, keepdims=True), ".1%", "Viridis"
        elif mode == "% columna":
            z, fmt, scale = t / t.sum(0, keepdims=True), ".1%", "Viridis"
        else:
            e = res["expected"] if res["expected"] is not None else t
            z, fmt, scale = (t - e) / np.sqrt(np.where(e > 0, e, 1)), ".1f", "RdBu_r"
        lim = np.nanmax(np.abs(z)) if mode.startswith("Residuos") else None
        fig = px.imshow(z, x=[str(v) for v in ct["cols"]], y=[str(v) for v in ct["rows"]], text_auto=fmt,
                        color_continuous_scale=scale, aspect="auto",
                        zmin=-lim if lim else None, zmax=lim, labels=dict(x=b, y=a, color=mode))
        show(fig, 120 + 32 * len(ct["rows"]))
        if mode.startswith("Residuos"):
            st.caption("Residuos > 2 (rojo) indican combinaciones más frecuentes de lo esperado bajo independencia; "
                       "< −2 (azul), menos frecuentes.")
    with tabs[1]:
        rp = t / t.sum(1, keepdims=True)
        fig = go.Figure([go.Bar(y=[str(v) for v in ct["rows"]], x=rp[:, j], name=str(cname), orientation="h")
                         for j, cname in enumerate(ct["cols"])])
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%", yaxis_autorange="reversed",
                          xaxis_title=f"composición de {b}", yaxis_title=a)
        show(fig, 120 + 32 * len(ct["rows"]))


def page_bivariate():
    st.header("🔀 Bivariado")
    sub = st.radio("Tipo de par", ["Numérica vs Numérica", "Numérica vs Categórica", "Categórica vs Categórica"],
                   horizontal=True)
    {"Numérica vs Numérica": num_num, "Numérica vs Categórica": num_cat,
     "Categórica vs Categórica": cat_cat}[sub]()


# --------------------------------------------------------------------------- #
# 5. Correlaciones
# --------------------------------------------------------------------------- #
def page_correlations():
    st.header("🧮 Correlaciones y asociaciones")
    sub = st.radio("Matriz", ["Numéricas", "Categóricas (V de Cramér)", "Mixtas (η)"], horizontal=True)
    if sub == "Numéricas":
        if not need(NUMS, 2, "numéricas"):
            return
        c1, c2 = st.columns([1, 2])
        method = c1.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
        cols = c2.multiselect("Variables", NUMS, NUMS[:30])
        if len(cols) < 2:
            return
        rows = min(test_n, 20_000) if method == "kendall" else max(test_n, 1_000_000)
        mat = cached("corr_matrix", KEY, df, cols, method, rows, seed)
        st.caption(f"Calculada sobre {min(rows, df.height):,} filas"
                   + (" (Kendall es O(n log n) por par: se usa una muestra menor)." if method == "kendall" else "."))
        show(px.imshow(mat, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                       text_auto=".2f" if len(cols) <= 15 else False, aspect="auto"), 150 + 30 * len(cols))
        thr = st.slider("Mostrar pares con |coef| ≥", 0.0, 1.0, 0.5, 0.05)
        tp = ec.top_pairs(mat, thr)
        st.dataframe(tp, hide_index=True, width="stretch")
        if tp.filter(pl.col("abs") >= 0.95).height:
            st.warning("Hay pares con |coef| ≥ 0.95: posible redundancia/multicolinealidad.")
    elif sub.startswith("Categóricas"):
        if not need(CATS, 2, "categóricas"):
            return
        cols = st.multiselect("Variables", CATS, CATS[:20])
        if len(cols) < 2:
            return
        mat = cached("cramers_matrix", KEY, df, cols, top_k, test_n, seed)
        show(px.imshow(mat, zmin=0, zmax=1, color_continuous_scale="Viridis", text_auto=".2f", aspect="auto"),
             150 + 30 * len(cols))
        st.caption(f"V de Cramér corregida, top-{top_k} categorías por variable, muestra de {min(test_n, df.height):,} filas.")
    else:
        if not (need(NUMS, 1, "numéricas") and need(CATS, 1, "categóricas")):
            return
        c1, c2 = st.columns(2)
        nc = c1.multiselect("Numéricas", NUMS, NUMS[:20])
        cc = c2.multiselect("Categóricas", CATS, CATS[:15])
        if not nc or not cc:
            return
        mat = cached("eta_matrix", KEY, df, nc, cc, top_k, test_n, seed)
        show(px.imshow(mat, zmin=0, zmax=1, color_continuous_scale="Viridis", text_auto=".2f", aspect="auto"),
             150 + 30 * len(nc))
        st.caption("Razón de correlación η: raíz de la fracción de varianza de la numérica explicada por la categórica.")


# --------------------------------------------------------------------------- #
# 6. Temporal
# --------------------------------------------------------------------------- #
def page_temporal():
    st.header("🕒 Temporal")
    if not need(DATES, 1, "de fecha/datetime"):
        st.caption("Si tu fecha viene como texto, conviértela antes (en CSV se intenta parsear automáticamente).")
        return
    c = st.columns(5)
    dcol = c[0].selectbox("Fecha", DATES)
    freqs = [f for f in FREQS if not (df.schema[dcol] == pl.Date and f == "hora")]
    freq = c[1].selectbox("Periodo", freqs, freqs.index("mes"))
    val = c[2].selectbox("Métrica", ["(conteo de filas)"] + NUMS)
    agg = c[3].selectbox("Agregación", list(ec.AGGS), disabled=val == "(conteo de filas)")
    by = c[4].selectbox("Separar por", ["(ninguno)"] + CATS)
    ts = cached("time_series", KEY, df, dcol, FREQS[freq], None if val.startswith("(") else val, agg,
                None if by == "(ninguno)" else by, 6)
    title = f"{'Filas' if val.startswith('(') else f'{agg} de {val}'} por {freq}"
    show(px.line(ts.to_pandas(), x="periodo", y="valor", color="serie" if "serie" in ts.columns else None,
                 markers=ts.height < 60, title=title), 450)
    st.caption("Agregado con todas las filas. Revisa huecos, cambios de nivel y estacionalidad: "
               "influyen en cómo partir train/test (split temporal).")


# --------------------------------------------------------------------------- #
# 7. AI Ready
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Evaluando preparación para IA…", max_entries=16)
def _ai(key: str, target, corr_thr, leak_thr, _df, _kinds, _prof, _nsum, _corr, _dups, _k, _n, _seed):
    return ec.ai_readiness(_df, _kinds, _prof, _nsum, _corr, _dups, target, _k, _n, _seed, corr_thr, leak_thr)


def get_corr():
    return cached("corr_matrix", KEY, df, NUMS, "pearson", max(test_n, 1_000_000), seed) if len(NUMS) > 1 else None


def get_ai(target: str | None, corr_thr: float = 0.95, leak_thr: float = 0.95):
    return _ai(f"{KEY}|{top_k}|{test_n}", target, corr_thr, leak_thr, df, kinds, get_profile(), get_nsum(),
               get_corr(), get_dups(), top_k, test_n, seed)


def page_ai():
    st.header("🤖 AI Ready")
    st.caption("Diagnóstico heurístico de qué tan lista está la tabla para entrenar un modelo, "
               "con acciones de preparación por columna.")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Variable objetivo (target)", ["(ninguna)"] + [c for c in kinds if kinds[c] in (ec.NUM, ec.CAT)])
    corr_thr = c2.slider("Umbral redundancia |r|", 0.8, 1.0, 0.95, 0.01)
    leak_thr = c3.slider("Umbral posible fuga", 0.8, 1.0, 0.95, 0.01)
    target = None if target.startswith("(") else target
    ai = get_ai(target, corr_thr, leak_thr)

    s = ai["score"]
    l, r = st.columns([1, 3])
    l.metric("Puntuación AI Ready", f"{s}/100")
    l.progress(s / 100)
    l.caption("🟢 ≥ 80 · 🟡 60–79 · 🔴 < 60")
    r.dataframe(ai["checks"], hide_index=True, width="stretch")

    st.subheader("Acciones por columna")
    st.dataframe(ai["actions"], hide_index=True, width="stretch")

    if ai["assoc"] is not None:
        st.subheader(f"Asociación con el target «{target}»")
        a = ai["assoc"].head(30).to_pandas()
        fig = px.bar(a, x="asociación", y="variable", color="métrica", orientation="h")
        fig.add_vline(x=leak_thr, line_dash="dash", line_color="red", annotation_text="posible fuga")
        fig.update_layout(yaxis_autorange="reversed", xaxis_range=[0, 1])
        show(fig, 80 + 26 * len(a))
        st.caption("Ranking univariado (no causal). Valores casi perfectos suelen ser fugas: variables que se "
                   "conocen después del evento que se quiere predecir.")

    st.subheader("Entregables")
    card = ec.data_card(df, kinds, get_profile(), get_nsum(), ai, ds_name)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Data card (JSON, contexto para LLM)", ec.to_json(card), f"{ds_name}_datacard.json",
                       "application/json", width="stretch")
    if d2.button("Preparar dataset limpio (Parquet)", width="stretch",
                 help="Elimina columnas marcadas 'Eliminar' y filas duplicadas."):
        clean = df.drop(ai["drop"]).unique(maintain_order=True)
        buf = io.BytesIO()
        clean.write_parquet(buf)
        d2.download_button(f"⬇️ {ds_name}_clean.parquet ({clean.height:,} filas)", buf.getvalue(),
                           f"{ds_name}_clean.parquet", width="stretch")
    with st.expander("Pipeline de preprocesamiento sugerido (scikit-learn)"):
        st.code(ec.sklearn_snippet(ai), language="python")


# --------------------------------------------------------------------------- #
# 8. Exportación
# --------------------------------------------------------------------------- #
def page_export():
    st.header("💾 Exportación")
    prof, nsum = get_profile(), get_nsum()
    corr = get_corr()
    pairs =ec.top_pairs(corr, 0.3) if corr is not None else None
    ai = get_ai(None)

    st.subheader("Informe")
    md = ec.markdown_report(ds_name, cached("overview", KEY, df, kinds), prof, nsum, pairs, ai)
    c = st.columns(2)
    c[0].download_button("⬇️ Informe EDA (Markdown)", md, f"{ds_name}_eda.md", "text/markdown", width="stretch")
    c[1].download_button("⬇️ Data card (JSON)", ec.to_json(ec.data_card(df, kinds, prof, nsum, ai, ds_name)),
                         f"{ds_name}_datacard.json", "application/json", width="stretch")

    st.subheader("Tablas")
    c = st.columns(3)
    c[0].download_button("⬇️ Perfil de columnas (CSV)", prof.write_csv(), f"{ds_name}_perfil.csv", width="stretch")
    if nsum.height:
        c[1].download_button("⬇️ Resumen numérico (CSV)", nsum.write_csv(), f"{ds_name}_numericas.csv", width="stretch")
    if corr is not None:
        c[2].download_button("⬇️ Matriz Pearson (CSV)", corr.to_csv(), f"{ds_name}_pearson.csv", width="stretch")

    st.subheader("Muestra de datos")
    c = st.columns(3)
    n = int(c[0].number_input("Filas de la muestra", 100, value=min(100_000, df.height), step=10_000))
    fmt = c[1].radio("Formato", ["Parquet", "CSV"], horizontal=True)
    if c[2].button("Preparar muestra", width="stretch"):
        s = ec.maybe_sample(df, n, seed)
        if fmt == "Parquet":
            buf = io.BytesIO()
            s.write_parquet(buf)
            data, ext = buf.getvalue(), "parquet"
        else:
            data, ext = s.write_csv(), "csv"
        st.download_button(f"⬇️ muestra_{s.height}.{ext}", data, f"{ds_name}_muestra.{ext}")

    with st.expander("Vista previa del informe"):
        st.markdown(md)


{SECTIONS[0]: page_profile, SECTIONS[1]: page_quality, SECTIONS[2]: page_univariate,
 SECTIONS[3]: page_bivariate, SECTIONS[4]: page_correlations, SECTIONS[5]: page_temporal,
 SECTIONS[6]: page_ai, SECTIONS[7]: page_export}[section]()
