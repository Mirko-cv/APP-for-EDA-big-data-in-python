"""Cálculos de EDA para tablas grandes (polars + numpy + scipy).

Regla general:
- Lo que se puede agregar (conteos, medias, cuantiles, histogramas, tablas de
  contingencia, ANOVA) se calcula sobre TODAS las filas: escala a millones.
- Lo que es costoso o solo sirve para dibujar (Kendall, Kruskal, violines,
  scatter, matrices de asociación) usa una MUESTRA controlada por el usuario.

Este módulo no depende de Streamlit: se puede usar desde un notebook.
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

NUM, CAT, DATE, OTHER = "numérica", "categórica", "fecha", "otra"
NULL_LABEL = "∅ (nulo)"
OTHER_LABEL = "Otros"
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

# Umbrales de alertas
TH_NULLS = 0.50
TH_NEAR_CONST = 0.99
TH_ID = 0.95
TH_HIGH_CARD = 50
TH_SKEW = 2.0
TH_OUTLIERS = 0.05


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _ext(name: str) -> str:
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def read_bytes(data: bytes, name: str, sep: str = ",", max_rows: int | None = None) -> pl.DataFrame:
    """Lee un archivo subido (csv/tsv/txt/parquet/jsonl)."""
    ext = _ext(name)
    buf = io.BytesIO(data)
    if ext == "parquet":
        return pl.read_parquet(buf, n_rows=max_rows)
    if ext in ("jsonl", "ndjson", "json"):
        df = pl.read_ndjson(buf)
        return df.head(max_rows) if max_rows else df
    if ext == "tsv":
        sep = "\t"
    kw = dict(separator=sep, try_parse_dates=True, n_rows=max_rows, encoding="utf8-lossy")
    try:
        return pl.read_csv(buf, infer_schema_length=10_000, **kw)
    except pl.exceptions.ComputeError:
        # Tipos mal inferidos en las primeras filas: reintenta mirando todo el archivo.
        buf.seek(0)
        return pl.read_csv(buf, infer_schema_length=None, **kw)


def read_path(path: str, sep: str = ",", max_rows: int | None = None) -> pl.DataFrame:
    """Lee desde disco con escaneo perezoso (admite globs, p. ej. data/*.parquet)."""
    ext = _ext(path)
    if ext == "parquet":
        lf = pl.scan_parquet(path)
    elif ext in ("jsonl", "ndjson"):
        lf = pl.scan_ndjson(path)
    else:
        sep = "\t" if ext == "tsv" else sep
        lf = pl.scan_csv(path, separator=sep, try_parse_dates=True,
                         infer_schema_length=10_000, encoding="utf8-lossy")
    if max_rows:
        lf = lf.head(max_rows)
    return lf.collect()


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Unifica tipos poco amigables: Categorical/Enum -> String, Decimal -> Float64."""
    casts = []
    for c, dt in df.schema.items():
        if isinstance(dt, (pl.Categorical, pl.Enum)) or dt == pl.Null:
            casts.append(pl.col(c).cast(pl.String))
        elif isinstance(dt, pl.Decimal):
            casts.append(pl.col(c).cast(pl.Float64))
    return df.with_columns(casts) if casts else df


def synthetic(n: int = 1_000_000, seed: int = 42) -> pl.DataFrame:
    """Dataset de clientes con problemas típicos: nulos, duplicados, ID, constante,
    columna redundante, alta cardinalidad, desbalance y una fuga de información."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["Norte", "Sur", "Este", "Oeste", "Centro", "Insular"], n,
                        p=[.25, .2, .2, .15, .15, .05])
    canal = rng.choice(["Web", "App", "Tienda", "Teléfono"], n, p=[.4, .3, .25, .05])
    edad = rng.integers(18, 80, n)
    ingreso = rng.lognormal(10 + 0.01 * (edad - 40), 0.6)
    mult = pd.Series(region).map({"Norte": 1.1, "Sur": .9, "Este": 1.0, "Oeste": 1.05,
                                  "Centro": 1.2, "Insular": .8}).to_numpy()
    gasto = np.clip(0.02 * ingreso * mult + rng.normal(0, 150, n) + 80 * (canal == "App"), 0, None)
    score = rng.normal(600, 80, n)
    promo = rng.random(n) < 0.3
    logit = -2.3 - 0.01 * (score - 600) + 0.6 * (canal == "Teléfono") - 0.3 * promo
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    fecha = np.datetime64("2024-01-01T00:00:00") + rng.integers(0, 730 * 86400, n).astype("timedelta64[s]")
    fecha = fecha.astype("datetime64[ms]")

    df = pl.DataFrame({
        "id_cliente": rng.permutation(n) + 100_000,
        "fecha_alta": fecha,
        "region": region,
        "canal": canal,
        "producto_id": (rng.zipf(1.4, n) % 800),
        "edad": edad,
        "ingreso": ingreso.round(2),
        "gasto_mensual": gasto.round(2),
        "gasto_anual": (gasto * 12 * (1 + rng.normal(0, 0.002, n))).round(2),
        "score_credito": score.round(1),
        "descuento": rng.uniform(0, 0.3, n).round(3),
        "promo": promo,
        "monto_reembolso": np.where(churn, gasto * rng.uniform(0.5, 1, n), 0).round(2),
        "version_esquema": np.full(n, "v1"),
        "churn": np.where(churn, "sí", "no"),
    })
    df = df.with_columns(
        pl.format("SKU-{}", pl.col("producto_id")).alias("producto"),
        pl.when(pl.lit(pl.Series(rng.random(n) < 0.04))).then(None).otherwise(pl.col("edad")).alias("edad"),
        pl.when(pl.lit(pl.Series(rng.random(n) < 0.35))).then(None).otherwise(pl.col("descuento")).alias("descuento"),
    ).drop("producto_id")
    dup_idx = rng.choice(n, n // 200, replace=False)
    return pl.concat([df, df.select(pl.all().gather(dup_idx))])


# --------------------------------------------------------------------------- #
# Tipos y utilidades
# --------------------------------------------------------------------------- #
def column_kinds(df: pl.DataFrame, as_cat=()) -> dict[str, str]:
    kinds = {}
    for c, dt in df.schema.items():
        if c in as_cat or dt == pl.Boolean or dt == pl.String:
            kinds[c] = CAT
        elif dt.is_numeric():
            kinds[c] = NUM
        elif dt == pl.Date or isinstance(dt, pl.Datetime):
            kinds[c] = DATE
        else:
            kinds[c] = OTHER
    return kinds


def low_card_int_cols(df: pl.DataFrame, max_levels: int = 10) -> list[str]:
    """Enteros con pocos valores distintos: candidatos a tratarse como categóricos."""
    ints = [c for c, dt in df.schema.items() if dt.is_integer()]
    if not ints:
        return []
    nu = df.select([pl.col(c).approx_n_unique() for c in ints]).row(0)
    return [c for c, u in zip(ints, nu) if u <= max_levels]


def fnum(c: str) -> pl.Expr:
    """Columna como Float64 con NaN/±inf convertidos a nulo."""
    x = pl.col(c).cast(pl.Float64)
    return pl.when(x.is_finite()).then(x).otherwise(None).alias(c)


def cat_expr(c: str) -> pl.Expr:
    return pl.col(c).cast(pl.String).fill_null(NULL_LABEL)


def lump_expr(df: pl.DataFrame, col: str, k: int, alias: str | None = None) -> pl.Expr:
    """Conserva las k categorías más frecuentes y agrupa el resto en 'Otros'."""
    s = cat_expr(col)
    top = (df.select(s.alias("v")).group_by("v").len("n")
             .sort("n", descending=True).head(k)["v"].to_list())
    return pl.when(s.is_in(top)).then(s).otherwise(pl.lit(OTHER_LABEL)).alias(alias or col)


def maybe_sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(n, seed=seed) if n and df.height > n else df


def strength(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "n/d"
    a = abs(v)
    for lim, lab in ((0.1, "despreciable"), (0.3, "débil"), (0.5, "moderada"), (0.7, "fuerte")):
        if a < lim:
            return lab
    return "muy fuerte"


# --------------------------------------------------------------------------- #
# Perfil y calidad
# --------------------------------------------------------------------------- #
def overview(df: pl.DataFrame, kinds: dict) -> dict:
    n, m = df.height, df.width
    nulls = sum(df.null_count().row(0)) if m else 0
    counts = {k: sum(1 for v in kinds.values() if v == k) for k in (NUM, CAT, DATE, OTHER)}
    return {"filas": n, "columnas": m, "memoria_mb": df.estimated_size("mb"),
            "pct_celdas_nulas": nulls / max(n * m, 1), **counts}


def profile(df: pl.DataFrame, kinds: dict) -> pl.DataFrame:
    """Una fila por columna: tipo, nulos, cardinalidad, valor dominante y alertas."""
    n = df.height
    exprs = []
    for c in df.columns:
        exprs.append(pl.col(c).null_count().alias(f"{c}§nulls"))
        if kinds[c] != OTHER:
            col = pl.col(c).to_physical() if kinds[c] == DATE else pl.col(c)
            exprs.append(col.drop_nulls().approx_n_unique().alias(f"{c}§nuniq"))
    agg = df.select(exprs).row(0, named=True)

    rows = []
    for c, dt in df.schema.items():
        k = kinds[c]
        nulls = agg[f"{c}§nulls"]
        nn = n - nulls
        nuniq = agg.get(f"{c}§nuniq")
        top_val, pct_top = None, None
        if nuniq is not None and nn and (k == CAT or nuniq <= 10_000):
            top = (df.select(pl.col(c)).drop_nulls().group_by(c).len("§n")
                     .sort("§n", descending=True).head(1))
            if top.height:
                top_val, pct_top = str(top[0, 0]), top[0, 1] / nn

        alerts = []
        if nn == 0:
            alerts.append("vacía")
        elif nuniq is not None and nuniq <= 1:
            alerts.append("constante")
        elif pct_top is not None and pct_top >= TH_NEAR_CONST:
            alerts.append("casi constante")
        if 0 < nulls / max(n, 1) and nulls / n >= TH_NULLS and nn:
            alerts.append("muchos nulos")
        is_id_type = k == CAT and dt == pl.String or (k == NUM and dt.is_integer())
        if nuniq and nn > 100 and is_id_type and nuniq / nn >= TH_ID:
            alerts.append("posible ID")
        elif k == CAT and nuniq and nuniq > TH_HIGH_CARD:
            alerts.append("alta cardinalidad")
        if k == NUM and dt.is_integer() and nuniq and 2 < nuniq <= 10:
            alerts.append("¿categórica?")

        rows.append({"columna": c, "dtype": str(dt), "clase": k, "nulos": nulls,
                     "pct_nulos": nulls / max(n, 1), "n_unicos": nuniq,
                     "pct_unicos": (nuniq / nn) if (nuniq is not None and nn) else None,
                     "top_valor": top_val, "pct_top": pct_top, "alertas": ", ".join(alerts)})
    return pl.DataFrame(rows, schema_overrides={"top_valor": pl.String, "pct_top": pl.Float64,
                                                "n_unicos": pl.Int64, "pct_unicos": pl.Float64})


def duplicate_rows(df: pl.DataFrame) -> int:
    """Filas sobrantes (las que se eliminarían con un drop_duplicates)."""
    try:
        return df.height - df.n_unique()
    except Exception:  # columnas anidadas no hasheables
        return -1


def null_patterns(df: pl.DataFrame, top: int = 10) -> pl.DataFrame | None:
    cols = [c for c, v in zip(df.columns, df.null_count().row(0)) if v > 0]
    if not cols:
        return None
    g = (df.select([pl.col(c).is_null() for c in cols]).group_by(cols).len("filas")
           .sort("filas", descending=True).head(top))
    patt = [" + ".join(c for c in cols if r[c]) or "(sin nulos)" for r in g.iter_rows(named=True)]
    return pl.DataFrame({"patrón de nulos": patt, "filas": g["filas"],
                         "pct": g["filas"] / df.height})


def nullity_corr(df: pl.DataFrame, max_rows: int, seed: int) -> pd.DataFrame | None:
    n = df.height
    cols = [c for c, v in zip(df.columns, df.null_count().row(0)) if 0 < v < n]
    if len(cols) < 2:
        return None
    d = maybe_sample(df.select(cols), max_rows, seed)
    return d.select([pl.col(c).is_null().cast(pl.Int8) for c in cols]).to_pandas().corr()


# --------------------------------------------------------------------------- #
# Univariado
# --------------------------------------------------------------------------- #
def numeric_summary(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    if not cols:
        return pl.DataFrame()
    exprs = []
    for c in cols:
        x = fnum(c)
        exprs += [x.count().alias(f"{c}§n"), x.mean().alias(f"{c}§media"), x.std().alias(f"{c}§desv"),
                  x.min().alias(f"{c}§min"), x.max().alias(f"{c}§max"),
                  x.skew().alias(f"{c}§asimetria"), x.kurtosis().alias(f"{c}§curtosis"),
                  (x == 0).cast(pl.Float64).mean().alias(f"{c}§pct_ceros"),
                  (x < 0).cast(pl.Float64).mean().alias(f"{c}§pct_negativos"),
                  pl.col(c).cast(pl.Float64).is_infinite().sum().alias(f"{c}§infinitos")]
        exprs += [x.quantile(q, "linear").alias(f"{c}§q{q}") for q in QS]
    r = df.select(exprs).row(0, named=True)

    out_exprs = []
    for c in cols:
        q1, q3 = r[f"{c}§q0.25"], r[f"{c}§q0.75"]
        if q1 is None:
            out_exprs.append(pl.lit(None, pl.Float64).alias(c))
            continue
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        x = fnum(c)
        out_exprs.append(((x < lo) | (x > hi)).cast(pl.Float64).mean().alias(c))
    outl = df.select(out_exprs).row(0, named=True)

    names = {0.01: "p1", 0.05: "p5", 0.25: "p25", 0.5: "mediana", 0.75: "p75", 0.95: "p95", 0.99: "p99"}
    rows = []
    for c in cols:
        row = {"variable": c}
        for key in ("n", "media", "desv", "min"):
            row[key] = r[f"{c}§{key}"]
        for q in QS:
            row[names[q]] = r[f"{c}§q{q}"]
        for key in ("max", "asimetria", "curtosis", "pct_ceros", "pct_negativos", "infinitos"):
            row[key] = r[f"{c}§{key}"]
        row["pct_outliers"] = outl[c]
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.exclude("variable", "n", "infinitos").cast(pl.Float64))


def histogram(df: pl.DataFrame, col: str, bins: int = 60, clip: bool = True, log: bool = False) -> dict | None:
    x = df.select(fnum(col)).drop_nulls().to_series().to_numpy()
    if log:
        x = np.log1p(x[x > -1])
    if x.size == 0:
        return None
    lo, hi = np.quantile(x, [0.005, 0.995]) if clip else (x.min(), x.max())
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    return {"counts": counts, "edges": edges, "n": int(x.size), "fuera_rango": int(x.size - counts.sum())}


def top_values(df: pl.DataFrame, col: str, k: int = 20) -> tuple[pl.DataFrame, int]:
    vc = (df.select(cat_expr(col).alias("valor")).group_by("valor").len("n")
            .sort("n", descending=True))
    n_levels = vc.height
    top = vc.head(k)
    rest = vc.slice(k)["n"].sum()
    if rest:
        top = pl.concat([top, pl.DataFrame({"valor": [OTHER_LABEL], "n": [rest]},
                                           schema=top.schema)])
    top = top.with_columns(pct=pl.col("n") / df.height).with_columns(pct_acum=pl.col("pct").cum_sum())
    return top, n_levels


# --------------------------------------------------------------------------- #
# Bivariado: numérica vs numérica
# --------------------------------------------------------------------------- #
def xy(df: pl.DataFrame, x: str, y: str) -> pl.DataFrame:
    return df.select(fnum(x).alias("x"), fnum(y).alias("y")).drop_nulls()


def _r_pvalue(r: float, n: int) -> float:
    if n < 3 or not np.isfinite(r):
        return float("nan")
    if abs(r) >= 1:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def pair_correlations(df: pl.DataFrame, x: str, y: str, kendall_n: int, seed: int) -> dict:
    d = xy(df, x, y)
    n = d.height
    if n < 3:
        return {"n": n}
    r = d.select(pl.corr("x", "y")).item()
    rho = d.select(pl.corr("x", "y", method="spearman")).item()
    s = maybe_sample(d, kendall_n, seed)
    tau, p_tau = stats.kendalltau(s["x"].to_numpy(), s["y"].to_numpy())
    rc = float(np.clip(r, -0.999999, 0.999999)) if r is not None and np.isfinite(r) else float("nan")
    z, se = math.atanh(rc) if np.isfinite(rc) else float("nan"), 1 / math.sqrt(max(n - 3, 1))
    m = d.select(pl.cov("x", "y").alias("cov"), pl.col("x").var().alias("vx"), pl.col("x").mean().alias("mx"),
                 pl.col("y").mean().alias("my")).row(0)
    slope = m[0] / m[1] if m[1] else float("nan")
    return {
        "n": n, "n_kendall": s.height,
        "pearson": r, "p_pearson": _r_pvalue(r, n),
        "pearson_ic95": (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)),
        "spearman": rho, "p_spearman": _r_pvalue(rho, n),
        "kendall": float(tau), "p_kendall": float(p_tau),
        "pendiente": slope, "intercepto": m[3] - slope * m[2],
    }


def hist2d(df: pl.DataFrame, x: str, y: str, bins: int = 80, clip: bool = True) -> dict | None:
    d = xy(df, x, y)
    if d.height == 0:
        return None
    if clip:
        q = d.select(pl.col("x").quantile(.005).alias("a"), pl.col("x").quantile(.995).alias("b"),
                     pl.col("y").quantile(.005).alias("c"), pl.col("y").quantile(.995).alias("d")).row(0)
    else:
        q = d.select(pl.col("x").min().alias("a"), pl.col("x").max().alias("b"),
                     pl.col("y").min().alias("c"), pl.col("y").max().alias("d")).row(0)
    xr = (q[0], q[1] if q[1] > q[0] else q[0] + 1)
    yr = (q[2], q[3] if q[3] > q[2] else q[2] + 1)

    def edges(col: str, r: tuple) -> np.ndarray:
        # Enteros con rango corto: un bin por valor, evita el rayado por aliasing.
        if df.schema[col].is_integer() and r[1] - r[0] + 1 <= bins:
            return np.arange(r[0] - 0.5, r[1] + 1.5)
        return np.linspace(r[0], r[1], bins + 1)

    xe, ye = edges(x, xr), edges(y, yr)
    H, xe, ye = np.histogram2d(d["x"].to_numpy(), d["y"].to_numpy(), bins=[xe, ye])
    xr, yr = (xe[0], xe[-1]), (ye[0], ye[-1])
    return {"H": H, "xc": (xe[:-1] + xe[1:]) / 2, "yc": (ye[:-1] + ye[1:]) / 2,
            "n": d.height, "en_rango": int(H.sum()), "xr": xr, "yr": yr}


def xy_arrays(df: pl.DataFrame, x: str, y: str, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    d = maybe_sample(xy(df, x, y), max_n, seed)
    return d["x"].to_numpy(), d["y"].to_numpy()


def scatter_sample(df: pl.DataFrame, x: str, y: str, n: int, seed: int,
                   color: str | None = None, k: int = 10) -> pd.DataFrame:
    cols = [fnum(x).alias("x"), fnum(y).alias("y")]
    if color:
        cols.append(lump_expr(df, color, k, "color"))
    d = df.select(cols).drop_nulls(["x", "y"])
    return maybe_sample(d, n, seed).to_pandas()


def binned_trend(df: pl.DataFrame, x: str, y: str, q: int = 20) -> pl.DataFrame:
    """Media y banda p10–p90 de Y por cuantiles de X: tendencia sin dibujar millones de puntos."""
    d = xy(df, x, y)
    return (d.with_columns(bin=pl.col("x").qcut(q, allow_duplicates=True))
              .group_by("bin")
              .agg(n=pl.len(), x=pl.col("x").mean(), y_media=pl.col("y").mean(),
                   y_mediana=pl.col("y").median(), y_p10=pl.col("y").quantile(.1),
                   y_p90=pl.col("y").quantile(.9))
              .sort("x").drop("bin"))


# --------------------------------------------------------------------------- #
# Bivariado: numérica vs categórica
# --------------------------------------------------------------------------- #
def _num_cat_frame(df: pl.DataFrame, num: str, cat: str, k: int) -> pl.DataFrame:
    return df.select(lump_expr(df, cat, k, "grupo"), fnum(num).alias("y")).drop_nulls("y")


def group_stats(df: pl.DataFrame, num: str, cat: str, k: int) -> pl.DataFrame:
    d = _num_cat_frame(df, num, cat, k)
    y = pl.col("y")
    g = d.group_by("grupo").agg(
        n=pl.len(), media=y.mean(), desv=y.std(), min=y.min(), p10=y.quantile(.1),
        p25=y.quantile(.25), mediana=y.median(), p75=y.quantile(.75), p90=y.quantile(.9), max=y.max())
    iqr = pl.col("p75") - pl.col("p25")
    return g.with_columns(
        lim_inf=pl.max_horizontal(pl.col("min"), pl.col("p25") - 1.5 * iqr),
        lim_sup=pl.min_horizontal(pl.col("max"), pl.col("p75") + 1.5 * iqr),
    ).sort("n", descending=True)


def anova_from_groups(g: pl.DataFrame) -> dict | None:
    """ANOVA de una vía exacto a partir de n, media y desviación por grupo (todas las filas)."""
    g = g.filter(pl.col("n") > 0)
    k = g.height
    n = int(g["n"].sum())
    if k < 2 or n <= k:
        return None
    ni, mi = g["n"].to_numpy().astype(float), g["media"].to_numpy()
    vi = np.nan_to_num(g["desv"].to_numpy().astype(float) ** 2)
    grand = (ni * mi).sum() / n
    ssb = (ni * (mi - grand) ** 2).sum()
    ssw = ((ni - 1) * vi).sum()
    dfb, dfw = k - 1, n - k
    F = (ssb / dfb) / (ssw / dfw) if ssw > 0 else float("inf")
    return {"F": F, "p": float(stats.f.sf(F, dfb, dfw)), "eta2": ssb / (ssb + ssw) if ssb + ssw else float("nan"),
            "omega2": (ssb - dfb * ssw / dfw) / (ssb + ssw + ssw / dfw) if ssw else float("nan"),
            "gl": (dfb, dfw), "n": n}


def stratified_sample(df: pl.DataFrame, num: str, cat: str, k: int, per_group: int, seed: int) -> pl.DataFrame:
    d = _num_cat_frame(df, num, cat, k)
    return (d.with_columns(r=pl.int_range(pl.len()).shuffle(seed=seed).over("grupo"))
              .filter(pl.col("r") < per_group).drop("r"))


def kruskal_test(df: pl.DataFrame, num: str, cat: str, k: int, per_group: int, seed: int) -> dict | None:
    s = stratified_sample(df, num, cat, k, per_group, seed)
    groups = [g["y"].to_numpy() for _, g in s.group_by("grupo")]
    groups = [a for a in groups if a.size >= 2]
    if len(groups) < 2:
        return None
    H, p = stats.kruskal(*groups)
    n = sum(a.size for a in groups)
    return {"H": float(H), "p": float(p), "epsilon2": float(H / (n - 1)), "n": n}


# --------------------------------------------------------------------------- #
# Bivariado: categórica vs categórica
# --------------------------------------------------------------------------- #
def crosstab(df: pl.DataFrame, a: str, b: str, k: int) -> dict:
    d = df.select(lump_expr(df, a, k, "a"), lump_expr(df, b, k, "b"))
    wide = d.group_by(["a", "b"]).len("n").pivot(on="b", index="a", values="n").fill_null(0)
    rows = wide["a"].to_list()
    cols = wide.columns[1:]
    t = wide.drop("a").to_numpy().astype(float)
    ro, co = np.argsort(-t.sum(1)), np.argsort(-t.sum(0))
    return {"table": t[ro][:, co], "rows": [rows[i] for i in ro], "cols": [cols[i] for i in co]}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cat_association(table: np.ndarray) -> dict:
    """Chi², V de Cramér con corrección de sesgo (Bergsma) y U de Theil en ambos sentidos."""
    r, c = table.shape
    n = table.sum()
    if r < 2 or c < 2 or n < 2:
        return {"cramers_v": float("nan"), "chi2": float("nan"), "p": float("nan"), "gl": 0,
                "expected": None, "theil_u_ab": float("nan"), "theil_u_ba": float("nan")}
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1))
    rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    denom = min(cc - 1, rc - 1)
    v = math.sqrt(phi2c / denom) if denom > 0 else float("nan")

    pj = table / n
    pa, pb = pj.sum(1), pj.sum(0)
    ha, hb = _entropy(pa), _entropy(pb)
    ha_b = sum(pb[j] * _entropy(pj[:, j] / pb[j]) for j in range(c) if pb[j] > 0)
    hb_a = sum(pa[i] * _entropy(pj[i, :] / pa[i]) for i in range(r) if pa[i] > 0)
    return {"cramers_v": v, "chi2": float(chi2), "p": float(p), "gl": int(dof), "expected": expected,
            "theil_u_ab": (ha - ha_b) / ha if ha else float("nan"),   # cuánto explica B a A
            "theil_u_ba": (hb - hb_a) / hb if hb else float("nan")}   # cuánto explica A a B


# --------------------------------------------------------------------------- #
# Correlaciones / asociaciones en matriz
# --------------------------------------------------------------------------- #
def corr_matrix(df: pl.DataFrame, cols: list[str], method: str, max_rows: int, seed: int) -> pd.DataFrame:
    d = maybe_sample(df.select([fnum(c) for c in cols]), max_rows, seed)
    return d.to_pandas().corr(method=method)


def top_pairs(mat: pd.DataFrame, thr: float = 0.0) -> pl.DataFrame:
    cols = list(mat.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = mat.iloc[i, j]
            if pd.notna(v) and abs(v) >= thr:
                rows.append({"var1": cols[i], "var2": cols[j], "coef": float(v), "abs": abs(float(v)),
                             "fuerza": strength(v)})
    if not rows:
        return pl.DataFrame(schema={"var1": pl.String, "var2": pl.String, "coef": pl.Float64,
                                    "abs": pl.Float64, "fuerza": pl.String})
    return pl.DataFrame(rows).sort("abs", descending=True)


def cramers_matrix(df: pl.DataFrame, cols: list[str], k: int, max_rows: int, seed: int) -> pd.DataFrame:
    d = maybe_sample(df.select(cols), max_rows, seed)
    m = np.eye(len(cols))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = cat_association(crosstab(d, cols[i], cols[j], k)["table"])["cramers_v"]
            m[i, j] = m[j, i] = v
    return pd.DataFrame(m, index=cols, columns=cols)


def _eta(d: pl.DataFrame, num: str, cat: str, k: int) -> float:
    sub = d.select(lump_expr(d, cat, k, "§g"), fnum(num).alias("y")).drop_nulls("y")
    if sub.height < 3:
        return float("nan")
    gm = sub["y"].mean()
    sst = sub.select(((pl.col("y") - gm) ** 2).sum()).item()
    g = sub.group_by("§g").agg(n=pl.len(), m=pl.col("y").mean())
    ssb = g.select((pl.col("n") * (pl.col("m") - gm) ** 2).sum()).item()
    return math.sqrt(ssb / sst) if sst > 0 else float("nan")


def eta_matrix(df: pl.DataFrame, num_cols: list[str], cat_cols: list[str], k: int,
               max_rows: int, seed: int) -> pd.DataFrame:
    """Razón de correlación η (0–1): cuánto de la varianza de la numérica explica la categórica."""
    d = maybe_sample(df.select(num_cols + cat_cols), max_rows, seed)
    m = [[_eta(d, nm, c, k) for c in cat_cols] for nm in num_cols]
    return pd.DataFrame(m, index=num_cols, columns=cat_cols)


# --------------------------------------------------------------------------- #
# Temporal
# --------------------------------------------------------------------------- #
AGGS = {"media": lambda e: e.mean(), "mediana": lambda e: e.median(), "suma": lambda e: e.sum(),
        "p90": lambda e: e.quantile(.9), "mín": lambda e: e.min(), "máx": lambda e: e.max()}


def time_series(df: pl.DataFrame, date_col: str, every: str, value: str | None = None,
                agg: str = "media", by: str | None = None, k: int = 5) -> pl.DataFrame:
    cols = [pl.col(date_col).dt.truncate(every).alias("periodo")]
    if by:
        cols.append(lump_expr(df, by, k, "serie"))
    if value:
        cols.append(fnum(value).alias("v"))
    d = df.select(cols).drop_nulls("periodo")
    keys = ["periodo"] + (["serie"] if by else [])
    e = pl.len() if value is None else AGGS[agg](pl.col("v"))
    return d.group_by(keys).agg(e.alias("valor")).sort(keys)


# --------------------------------------------------------------------------- #
# AI Ready
# --------------------------------------------------------------------------- #
def target_association(df: pl.DataFrame, target: str, kinds: dict, k: int,
                       max_rows: int, seed: int) -> pl.DataFrame:
    """Asociación de cada variable con el target, en escala 0–1.
    Target binario + feature numérica: Gini univariado = |2·AUC − 1| (detecta fugas
    del tipo 'solo vale > 0 cuando y = 1' que η no ve bien)."""
    d = maybe_sample(df, max_rows, seed).filter(pl.col(target).is_not_null())
    levels = d[target].unique().to_list() if kinds[target] == CAT else []
    rows = []
    for c, kc in kinds.items():
        if c == target or kc not in (NUM, CAT):
            continue
        if len(levels) == 2 and kc == NUM:
            sub = d.select(fnum(c).alias("x"), (pl.col(target) == levels[1]).alias("t")).drop_nulls()
            x1, x0 = sub.filter("t")["x"].to_numpy(), sub.filter(~pl.col("t"))["x"].to_numpy()
            if x1.size and x0.size:
                auc = stats.mannwhitneyu(x1, x0).statistic / (x1.size * x0.size)
                v = abs(2 * auc - 1)
            else:
                v = float("nan")
            met = "Gini (|2·AUC−1|)"
        elif kinds[target] == NUM:
            if kc == NUM:
                dd = xy(d, c, target)
                v, met = (abs(dd.select(pl.corr("x", "y")).item() or 0) if dd.height > 2 else float("nan")), "|Pearson|"
            else:
                v, met = _eta(d, target, c, k), "η"
        else:
            if kc == NUM:
                v, met = _eta(d, c, target, k), "η"
            else:
                v, met = cat_association(crosstab(d, target, c, k)["table"])["cramers_v"], "V de Cramér"
        rows.append({"variable": c, "métrica": met, "asociación": float(v) if v is not None else float("nan")})
    if not rows:
        return pl.DataFrame(schema={"variable": pl.String, "métrica": pl.String, "asociación": pl.Float64})
    return (pl.DataFrame(rows).with_columns(pl.col("asociación").fill_nan(None))
              .sort("asociación", descending=True, nulls_last=True))


def ai_readiness(df: pl.DataFrame, kinds: dict, prof: pl.DataFrame, nsum: pl.DataFrame,
                 corr: pd.DataFrame | None, dup_rows: int, target: str | None, k: int,
                 max_rows: int, seed: int, corr_thr: float = 0.95, leak_thr: float = 0.95) -> dict:
    n = df.height
    checks, penalty = [], 0.0
    P = {r["columna"]: r for r in prof.to_dicts()}
    S = {r["variable"]: r for r in nsum.to_dicts()} if nsum.height else {}

    # Duplicados
    if dup_rows > 0:
        pct = dup_rows / n
        penalty += 10 if pct > 0.01 else 3
        checks.append(("⚠️", "Filas duplicadas", f"{dup_rows:,} filas ({pct:.2%}). Deduplicar antes de partir train/test."))
    elif dup_rows == 0:
        checks.append(("✅", "Filas duplicadas", "Sin duplicados exactos."))

    # Tamaño
    if n < 1_000:
        penalty += 5
        checks.append(("⚠️", "Tamaño", f"Solo {n:,} filas: riesgo de sobreajuste."))

    # Nulos globales
    avg_null = float(prof["pct_nulos"].mean()) if prof.height else 0
    penalty += 15 * avg_null
    lvl = "✅" if avg_null < 0.05 else "⚠️"
    checks.append((lvl, "Nulos", f"Promedio de nulos por columna: {avg_null:.1%}."))

    # Redundancia
    redundant = {}
    if corr is not None and corr.shape[0] > 1:
        tp = top_pairs(corr, corr_thr)
        if target:
            tp = tp.filter((pl.col("var1") != target) & (pl.col("var2") != target))
        for r in tp.iter_rows(named=True):
            redundant.setdefault(r["var2"], (r["var1"], r["coef"]))
        if tp.height:
            penalty += min(10, 2 * tp.height)
            checks.append(("⚠️", "Multicolinealidad",
                           f"{tp.height} pares con |r| ≥ {corr_thr}: " +
                           ", ".join(f"{a}~{b}" for a, b in zip(tp["var1"][:5], tp["var2"][:5]))))
        else:
            checks.append(("✅", "Multicolinealidad", f"Ningún par con |r| ≥ {corr_thr}."))

    # Target
    assoc, leaks = None, set()
    if target:
        assoc = target_association(df, target, kinds, k, max_rows, seed)
        leaks = set(assoc.filter(pl.col("asociación") >= leak_thr)["variable"].to_list())
        if leaks:
            penalty += min(20, 10 * len(leaks))
            checks.append(("❌", "Posible fuga de información",
                           f"Asociación ≥ {leak_thr} con el target: {', '.join(sorted(leaks))}."))
        else:
            checks.append(("✅", "Fuga de información", "Ninguna variable con asociación casi perfecta con el target."))
        if P[target]["nulos"]:
            penalty += 5
            checks.append(("⚠️", "Target con nulos", f"{P[target]['nulos']:,} filas sin target: excluirlas del entrenamiento."))
        if kinds[target] == CAT:
            tv, nlev = top_values(df, target, 50)
            minority = float(tv["pct"].min())
            if nlev > 50:
                checks.append(("⚠️", "Target", f"{nlev} clases: ¿regresión o agrupar clases?"))
            if minority < 0.05:
                penalty += 10
                checks.append(("⚠️", "Desbalance", f"Clase minoritaria = {minority:.2%}. Usar estratificación, "
                                                   "métricas PR-AUC/F1 y pesos de clase."))
            else:
                checks.append(("✅", "Balance de clases", f"Clase minoritaria = {minority:.2%}."))
        else:
            sk = S.get(target, {}).get("asimetria")
            if sk is not None and abs(sk) > TH_SKEW:
                checks.append(("⚠️", "Target sesgado", f"Asimetría {sk:.2f}: considerar log del target o métricas robustas."))

    # Acciones por columna
    actions, drop = [], []
    groups = {"num": [], "num_skew": [], "cat_ohe": [], "cat_high": [], "date": []}
    for c, kc in kinds.items():
        if c == target:
            continue
        p, acts, status = P[c], [], "OK"
        al = p["alertas"]
        if "vacía" in al or "constante" in al and "casi" not in al:
            acts, status = ["eliminar (sin información)"], "Eliminar"
        elif "posible ID" in al:
            acts, status = ["excluir como feature (identificador); útil para joins/deduplicar"], "Eliminar"
        elif c in leaks:
            acts, status = ["revisar: posible fuga de información (¿se conoce en el momento de predecir?)"], "Eliminar"
        else:
            if c in redundant:
                o, r = redundant[c]
                acts.append(f"redundante con {o} (r={r:.3f}); conservar solo una")
                status = "Revisar"
            if p["pct_nulos"] >= TH_NULLS:
                acts.append(f"{p['pct_nulos']:.0%} nulos: eliminar o imputar + indicador")
                status = "Revisar"
            elif p["pct_nulos"] > 0:
                acts.append("imputar (" + ("mediana" if kc == NUM else "moda / categoría 'nulo'") + ") + indicador")
            if "casi constante" in al:
                acts.append("casi constante: evaluar eliminar")
                status = "Revisar"
            if kc == NUM:
                s = S.get(c, {})
                if s.get("asimetria") is not None and abs(s["asimetria"]) > TH_SKEW:
                    acts.append("transformar (log1p / Yeo-Johnson)")
                    groups["num_skew"].append(c)
                else:
                    groups["num"].append(c)
                if (s.get("pct_outliers") or 0) > TH_OUTLIERS:
                    acts.append(f"{s['pct_outliers']:.1%} outliers: escalado robusto / winsorizar")
                else:
                    acts.append("escalar (estándar)")
            elif kc == CAT:
                nu = p["n_unicos"] or 0
                if nu <= 15:
                    acts.append("one-hot")
                    groups["cat_ohe"].append(c)
                else:
                    acts.append("target / frequency encoding o agrupar categorías raras")
                    groups["cat_high"].append(c)
            elif kc == DATE:
                acts.append("extraer año/mes/día-semana/hora; usar para split temporal")
                groups["date"].append(c)
            else:
                acts, status = ["tipo anidado/binario: aplanar o excluir"], "Revisar"
        if status == "Eliminar":
            drop.append(c)
        actions.append({"columna": c, "clase": kc, "estado": status, "acciones": "; ".join(acts)})

    penalty += min(15, 3 * len(drop))
    if drop:
        checks.append(("⚠️", "Columnas a eliminar", ", ".join(drop)))
    score = int(max(0, round(100 - penalty)))
    return {"score": score, "checks": pl.DataFrame(checks, schema=["nivel", "aspecto", "detalle"], orient="row"),
            "actions": pl.DataFrame(actions), "assoc": assoc, "drop": drop, "groups": groups,
            "target": target}


def data_card(df: pl.DataFrame, kinds: dict, prof: pl.DataFrame, nsum: pl.DataFrame,
              ai: dict, name: str, top_k: int = 5) -> dict:
    """Ficha del dataset en JSON: contexto compacto para un LLM o para documentación."""
    S = {r["variable"]: r for r in nsum.to_dicts()} if nsum.height else {}
    A = {r["columna"]: r for r in ai["actions"].to_dicts()}
    cols = []
    for r in prof.to_dicts():
        c = r["columna"]
        item = {"nombre": c, "dtype": r["dtype"], "clase": r["clase"],
                "pct_nulos": round(r["pct_nulos"], 4), "n_unicos_aprox": r["n_unicos"],
                "alertas": [a for a in r["alertas"].split(", ") if a]}
        if c in S:
            item["estadisticos"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                    for k, v in S[c].items() if k != "variable"}
        if r["clase"] == CAT:
            tv, _ = top_values(df, c, top_k)
            item["top_valores"] = [{"valor": v, "pct": round(p, 4)} for v, p in zip(tv["valor"], tv["pct"])]
        if r["clase"] == DATE:
            mn, mx = df.select(pl.col(c).min().alias("a"), pl.col(c).max().alias("b")).row(0)
            item["rango"] = [str(mn), str(mx)]
        if c in A:
            item["recomendacion"] = {"estado": A[c]["estado"], "acciones": A[c]["acciones"]}
        cols.append(item)
    return {
        "dataset": name, "generado": datetime.now().isoformat(timespec="seconds"),
        "filas": df.height, "columnas": df.width, "target": ai["target"],
        "ai_ready_score": ai["score"],
        "chequeos": [{"nivel": a, "aspecto": b, "detalle": c} for a, b, c in ai["checks"].iter_rows()],
        "columnas_detalle": cols,
    }


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        return str(o)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)


def sklearn_snippet(ai: dict) -> str:
    g, t = ai["groups"], ai["target"]
    return f'''import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, TargetEncoder

TARGET   = {t!r}
DROP     = {ai["drop"]!r}
NUM      = {g["num"]!r}
NUM_SKEW = {g["num_skew"]!r}
CAT_OHE  = {g["cat_ohe"]!r}
CAT_HIGH = {g["cat_high"]!r}
DATES    = {g["date"]!r}  # derivar features (año, mes, día de semana...) antes del pipeline

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("esc", RobustScaler())])
skew_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("yj", PowerTransformer(method="yeo-johnson"))])
ohe_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__nulo__")),
                     ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01))])
high_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="__nulo__")),
                      ("te", TargetEncoder())])

preprocesador = ColumnTransformer([
    ("num", num_pipe, NUM),
    ("skew", skew_pipe, NUM_SKEW),
    ("ohe", ohe_pipe, CAT_OHE),
    ("high", high_pipe, CAT_HIGH),
], remainder="drop")

# df = pd.read_parquet("datos.parquet").drop(columns=DROP).drop_duplicates()
# X, y = df.drop(columns=[TARGET]), df[TARGET]
# X_t = preprocesador.fit_transform(X, y)   # ajustar SOLO con train
'''


# --------------------------------------------------------------------------- #
# Informe Markdown
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.4g}"
    return str(v).replace("|", "\\|")


def md_table(df: pl.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.height == 0:
        return "_(sin datos)_\n"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.head(max_rows).iter_rows()]
    return "\n".join(lines) + "\n"


def markdown_report(name: str, ov: dict, prof: pl.DataFrame, nsum: pl.DataFrame,
                    pairs: pl.DataFrame | None, ai: dict | None) -> str:
    out = [f"# Informe EDA — {name}", f"_Generado: {datetime.now():%Y-%m-%d %H:%M}_\n",
           "## Resumen",
           f"- Filas: **{ov['filas']:,}** · Columnas: **{ov['columnas']}** · Memoria: {ov['memoria_mb']:.1f} MB",
           f"- Numéricas: {ov[NUM]} · Categóricas: {ov[CAT]} · Fechas: {ov[DATE]} · Otras: {ov[OTHER]}",
           f"- Celdas nulas: {ov['pct_celdas_nulas']:.2%}\n",
           "## Perfil de columnas", md_table(prof),
           "## Resumen numérico", md_table(nsum)]
    if pairs is not None:
        out += ["## Pares más correlacionados (Pearson)", md_table(pairs.head(20))]
    if ai:
        out += [f"## AI Ready — puntuación {ai['score']}/100", md_table(ai["checks"]),
                "### Acciones por columna", md_table(ai["actions"])]
    return "\n".join(out)
