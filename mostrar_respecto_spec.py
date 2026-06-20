import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# ---------------------------------------------------------
# Funciones auxiliares generales
# ---------------------------------------------------------
def columna_o_cero(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve la columna si existe. Si no existe, devuelve ceros."""
    if col in df.columns:
        return df[col]
    return pd.Series(0.0, index=df.index)


def normalizar_nombre_columna(nombre: str) -> str:
    """Normaliza nombres de columnas para detectar aliases con más facilidad."""
    nombre = str(nombre).strip().lower()
    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    nombre = re.sub(r"[^a-z0-9]+", "_", nombre)
    nombre = re.sub(r"_+", "_", nombre).strip("_")
    return nombre


def convertir_a_numerico(serie: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérica aceptando decimal con punto o coma.
    Ejemplos válidos: 123.45, 123,45, "1 234,45", "45%".
    """
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    s = serie.astype(str).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    mask_coma_decimal = s.str.contains(",", regex=False) & ~s.str.contains(".", regex=False)
    s.loc[mask_coma_decimal] = s.loc[mask_coma_decimal].str.replace(",", ".", regex=False)

    mask_miles_coma = s.str.contains(".", regex=False) & s.str.contains(",", regex=False)
    s.loc[mask_miles_coma] = (
        s.loc[mask_miles_coma]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(s, errors="coerce")


def leer_csv_flexible(path: str) -> pd.DataFrame:
    """Lee CSV intentando detectar separador automáticamente."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)


def buscar_columna_por_alias(
    df: pd.DataFrame,
    aliases: List[str],
    obligatoria: bool = True,
    descripcion: str = "columna",
) -> Optional[str]:
    """Busca una columna usando una lista de aliases normalizados."""
    normalizadas = {normalizar_nombre_columna(c): c for c in df.columns}
    aliases_norm = [normalizar_nombre_columna(a) for a in aliases]

    for alias in aliases_norm:
        if alias in normalizadas:
            return normalizadas[alias]

    for alias in aliases_norm:
        for col_norm, col_original in normalizadas.items():
            if alias and alias in col_norm:
                return col_original

    if obligatoria:
        raise ValueError(
            f"No se ha encontrado la {descripcion}. Columnas disponibles: {list(df.columns)}"
        )

    return None


def formatear_float(v: float, decimales: int = 4) -> str:
    if pd.isna(v):
        return "no disponible"
    return f"{v:.{decimales}f}"


# ---------------------------------------------------------
# Cargar SPEC
# ---------------------------------------------------------
def cargar_spec(
    spec_path: str,
    spec_util_col: Optional[str] = None,
    spec_power_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carga el CSV de SPECpower.

    Debe contener una columna de utilización/carga y una columna de potencia.
    El script intenta detectar nombres habituales automáticamente.
    """
    df = leer_csv_flexible(spec_path)

    util_aliases = [
        "target_load",
        "target_load_pct",
        "target_load_percent",
        "target load",
        "target load %",
        "actual_load",
        "actual_load_pct",
        "actual load",
        "ssj_load",
        "load",
        "load_pct",
        "load %",
        "utilization",
        "utilisation",
        "utilizacion",
        "utilización",
        "util_pct",
        "utilizacion_pct",
        "utilización_pct",
        "percent_utilization",
        "pct_utilization",
    ]

    power_aliases = [
        "average_active_power",
        "average active power",
        "avg_active_power",
        "avg active power",
        "active_power",
        "active power",
        "avg_power",
        "average_power",
        "avg power",
        "average power",
        "power",
        "power_w",
        "watts",
        "watt",
        "potencia",
        "potencia_w",
        "consumo",
        "consumo_w",
    ]

    if spec_util_col is None:
        spec_util_col = buscar_columna_por_alias(
            df,
            util_aliases,
            obligatoria=True,
            descripcion="columna de utilización/carga de SPEC",
        )

    if spec_power_col is None:
        spec_power_col = buscar_columna_por_alias(
            df,
            power_aliases,
            obligatoria=True,
            descripcion="columna de potencia de SPEC",
        )

    if spec_util_col not in df.columns:
        raise ValueError(f"La columna SPEC de utilización '{spec_util_col}' no existe")

    if spec_power_col not in df.columns:
        raise ValueError(f"La columna SPEC de potencia '{spec_power_col}' no existe")

    out = pd.DataFrame()
    out["utilizacion_pct"] = convertir_a_numerico(df[spec_util_col])
    out["spec_power_W"] = convertir_a_numerico(df[spec_power_col])

    out = out.dropna(subset=["utilizacion_pct", "spec_power_W"])

    if len(out) > 0 and out["utilizacion_pct"].max() <= 1.5:
        out["utilizacion_pct"] = out["utilizacion_pct"] * 100

    out["utilizacion_pct"] = out["utilizacion_pct"].round(6)

    out = (
        out.sort_values("utilizacion_pct")
        .drop_duplicates("utilizacion_pct", keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("El CSV de SPEC no contiene filas válidas de utilización/potencia")

    return out


# ---------------------------------------------------------
# Cargar CSV de MintEDGE
# ---------------------------------------------------------
def detectar_columna_utilizacion(
    df: pd.DataFrame,
    util_col: Optional[str] = None,
) -> Tuple[pd.Series, str]:
    """
    Detecta la utilización media de servidores.

    Prioridad:
    1. Columna indicada por --util-col.
    2. Columnas server_util_* generadas por MintEDGE.
    3. Aliases habituales de utilización media.
    """
    if util_col is not None:
        if util_col not in df.columns:
            raise ValueError(f"La columna de utilización '{util_col}' no existe")

        s = convertir_a_numerico(df[util_col])
        origen = util_col

    else:
        util_cols = [c for c in df.columns if str(c).startswith("server_util_")]

        if util_cols:
            util_df = df[util_cols].apply(convertir_a_numerico)
            s = util_df.mean(axis=1)
            origen = "media de columnas server_util_*"

        else:
            aliases = [
                "utilizacion_media_servidores",
                "utilización_media_servidores",
                "server_utilization",
                "server_utilization_mean",
                "mean_server_utilization",
                "avg_server_utilization",
                "utilization",
                "utilizacion",
                "utilización",
                "util_pct",
            ]

            col = buscar_columna_por_alias(
                df,
                aliases,
                obligatoria=True,
                descripcion="columna de utilización del CSV de MintEDGE",
            )

            s = convertir_a_numerico(df[col])
            origen = col

    if s.dropna().max() <= 1.5:
        s = s * 100

    return s.clip(lower=0, upper=100), origen


def calcular_potencia_mintedge(
    df: pd.DataFrame,
    power_mode: str,
    power_col: Optional[str] = None,
) -> Tuple[pd.Series, str]:
    """
    Calcula la potencia que se compara contra SPEC.

    power_mode:
    - servers: dynamic_W_servers + idle_W_servers. Recomendado para SPECpower.
    - total: dynamic_W_servers + idle_W_servers + W_links.
    - dynamic: solo dynamic_W_servers.
    - custom: usa --power-col.
    """
    if power_mode == "custom":
        if not power_col:
            raise ValueError("Con --power-mode custom debes indicar --power-col")

        if power_col not in df.columns:
            raise ValueError(f"La columna de potencia '{power_col}' no existe")

        return convertir_a_numerico(df[power_col]), power_col

    dynamic = convertir_a_numerico(columna_o_cero(df, "dynamic_W_servers"))
    idle = convertir_a_numerico(columna_o_cero(df, "idle_W_servers"))
    links = convertir_a_numerico(columna_o_cero(df, "W_links"))

    if power_mode == "servers":
        return dynamic + idle, "dynamic_W_servers + idle_W_servers"

    if power_mode == "total":
        return dynamic + idle + links, "dynamic_W_servers + idle_W_servers + W_links"

    if power_mode == "dynamic":
        if "dynamic_W_servers" not in df.columns:
            raise ValueError("No existe la columna dynamic_W_servers")

        return dynamic, "dynamic_W_servers"

    raise ValueError(f"power_mode no reconocido: {power_mode}")


def cargar_mintedge(
    csv_path: str,
    label: str,
    power_mode: str,
    power_col: Optional[str] = None,
    util_col: Optional[str] = None,
    tiempo_sin_arranque: float = 0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = leer_csv_flexible(csv_path)

    if "time" not in df.columns:
        raise ValueError(f"El archivo {csv_path} no tiene columna 'time'")

    df["time"] = convertir_a_numerico(df["time"])

    df = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )

    if tiempo_sin_arranque > 0:
        df = df[df["time"] > tiempo_sin_arranque].copy()

    if df.empty:
        raise ValueError(
            f"El archivo {csv_path} no tiene datos después de aplicar "
            f"--sin-arranque {tiempo_sin_arranque}"
        )

    df["utilizacion_pct"], origen_util = detectar_columna_utilizacion(df, util_col)
    df["power_W"], origen_power = calcular_potencia_mintedge(df, power_mode, power_col)
    df["modelo"] = label

    df = df.dropna(subset=["utilizacion_pct", "power_W"])

    if df.empty:
        raise ValueError(f"El archivo {csv_path} no contiene filas válidas de utilización/potencia")

    meta = {
        "path": csv_path,
        "label": label,
        "origen_utilizacion": origen_util,
        "origen_potencia": origen_power,
    }

    return df, meta


# ---------------------------------------------------------
# Comparación contra SPEC
# ---------------------------------------------------------
def preparar_puntos_modelo(
    df: pd.DataFrame,
    spec_utils: np.ndarray,
    metodo: str = "ventana",
    ventana: float = 5.0,
) -> pd.DataFrame:
    """
    Obtiene un punto del modelo para cada utilización publicada por SPEC.

    Importante:
    - Si el nivel SPEC está fuera del rango de utilización medido por MintEDGE,
      no se extrapola y se marca como no medido.
    - Dentro del rango medido, usa ventana o interpolación.
    """
    puntos = []

    base = df[["utilizacion_pct", "power_W"]].dropna().sort_values("utilizacion_pct")

    if base.empty:
        raise ValueError("No hay datos válidos para preparar puntos del modelo")

    curva = (
        base.groupby("utilizacion_pct", as_index=False)["power_W"]
        .mean()
        .sort_values("utilizacion_pct")
    )

    x = curva["utilizacion_pct"].to_numpy(dtype=float)
    y = curva["power_W"].to_numpy(dtype=float)

    util_min = float(x.min())
    util_max = float(x.max())

    for u in spec_utils:
        u = float(u)

        # No añadir puntos SPEC fuera del rango realmente medido por MintEDGE
        if u < util_min or u > util_max:
            puntos.append(
                {
                    "utilizacion_pct": u,
                    "modelo_power_W": np.nan,
                    "n_muestras_ventana": 0,
                    "origen_punto_modelo": "no_medido_fuera_rango",
                }
            )
            continue

        filas = base[
            (base["utilizacion_pct"] >= u - ventana)
            & (base["utilizacion_pct"] <= u + ventana)
        ]

        if metodo == "ventana" and not filas.empty:
            modelo_power = filas["power_W"].mean()
            n_muestras = len(filas)
            origen = "media_ventana"

        else:
            if len(x) == 1:
                modelo_power = float(y[0])
            else:
                modelo_power = float(np.interp(u, x, y))

            n_muestras = 0
            origen = "interpolado"

        puntos.append(
            {
                "utilizacion_pct": u,
                "modelo_power_W": modelo_power,
                "n_muestras_ventana": n_muestras,
                "origen_punto_modelo": origen,
            }
        )

    return pd.DataFrame(puntos)

def comparar_con_spec(
    spec_df: pd.DataFrame,
    modelos: Dict[str, pd.DataFrame],
    metodo: str,
    ventana: float,
) -> pd.DataFrame:
    comparaciones = []
    spec_utils = spec_df["utilizacion_pct"].to_numpy(dtype=float)

    for label, df_modelo in modelos.items():
        puntos = preparar_puntos_modelo(
            df_modelo,
            spec_utils,
            metodo=metodo,
            ventana=ventana,
        )

        comp = spec_df.merge(puntos, on="utilizacion_pct", how="left")
        comp["modelo"] = label

        comp["error_W"] = comp["modelo_power_W"] - comp["spec_power_W"]
        comp["error_abs_W"] = comp["error_W"].abs()

        comp["error_pct"] = (
            comp["error_W"] / comp["spec_power_W"].replace(0, np.nan)
        ) * 100

        comp["error_abs_pct"] = comp["error_pct"].abs()

        comp["potencia_modelo_sobre_SPEC_pct"] = (
            comp["modelo_power_W"] / comp["spec_power_W"].replace(0, np.nan)
        ) * 100

        comp["diferencia_respecto_SPEC_pct"] = (
            comp["potencia_modelo_sobre_SPEC_pct"] - 100
        )

        comparaciones.append(comp)

    return pd.concat(comparaciones, ignore_index=True)

def calcular_metricas_error(comp: pd.DataFrame) -> pd.DataFrame:
    filas = []

    for modelo, g in comp.groupby("modelo"):
        validos = g.dropna(subset=["modelo_power_W", "spec_power_W", "error_W"])

        if validos.empty:
            filas.append(
                {
                    "modelo": modelo,
                    "MAE_W": np.nan,
                    "RMSE_W": np.nan,
                    "MAPE_pct": np.nan,
                    "sesgo_medio_W": np.nan,
                    "max_error_abs_W": np.nan,
                    "max_error_abs_pct": np.nan,
                    "potencia_media_sobre_SPEC_pct": np.nan,
                    "puntos_comparados": 0,
                }
            )
            continue

        filas.append(
            {
                "modelo": modelo,
                "MAE_W": validos["error_abs_W"].mean(),
                "RMSE_W": float(np.sqrt(np.mean(np.square(validos["error_W"])))),
                "MAPE_pct": validos["error_abs_pct"].mean(),
                "sesgo_medio_W": validos["error_W"].mean(),
                "max_error_abs_W": validos["error_abs_W"].max(),
                "max_error_abs_pct": validos["error_abs_pct"].max(),
                "potencia_media_sobre_SPEC_pct": validos[
                    "potencia_modelo_sobre_SPEC_pct"
                ].mean(),
                "puntos_comparados": len(validos),
            }
        )

    return pd.DataFrame(filas)

# ---------------------------------------------------------
# SPEC temporal exacto
# ---------------------------------------------------------
def obtener_spec_esperado_temporal(
    df_modelo: pd.DataFrame,
    spec_df: pd.DataFrame,
    tolerancia: float = 0.01,
) -> pd.DataFrame:
    """
    Añade SPEC esperado cuando la utilización está cerca de un punto SPEC.

    Ejemplo con tolerancia=1:
    - SPEC 30% acepta utilización entre 29% y 31%.
    - SPEC 0% acepta entre 0% y 1%.
    - SPEC 100% acepta entre 99% y 100%.

    Si la utilización no cae dentro de ningún rango permitido, no se dibuja punto.
    """
    if spec_df.empty:
        raise ValueError("SPEC está vacío")

    out = df_modelo.copy()

    spec_ordenado = (
        spec_df[["utilizacion_pct", "spec_power_W"]]
        .dropna()
        .sort_values("utilizacion_pct")
    )

    xs = spec_ordenado["utilizacion_pct"].to_numpy(dtype=float)
    ys = spec_ordenado["spec_power_W"].to_numpy(dtype=float)

    util = out["utilizacion_pct"].to_numpy(dtype=float)

    esperado = np.full(len(out), np.nan)
    nivel = np.full(len(out), np.nan)

    tolerancia = max(float(tolerancia), 0.0)

    for i, u in enumerate(util):
        diferencias = np.abs(xs - u)
        idx = int(np.argmin(diferencias))

        nivel_spec = float(xs[idx])

        limite_inferior = max(0.0, nivel_spec - tolerancia)
        limite_superior = min(100.0, nivel_spec + tolerancia)

        if limite_inferior <= u <= limite_superior:
            esperado[i] = ys[idx]
            nivel[i] = nivel_spec

    out["spec_esperado_temporal_W"] = esperado
    out["nivel_spec_temporal_pct"] = nivel

    return out

def obtener_eventos_cambio_spec_temporal(
    df_temporal: pd.DataFrame,
    modelo: str,
    incluir_primer_punto: bool = False,
) -> pd.DataFrame:
    """
    Detecta puntos donde la utilización coincide con un nivel SPEC:
    0, 10, 20, ..., 100.

    Para las anotaciones se evita repetir muchas veces el mismo nivel.
    """
    cols = [
        "time",
        "utilizacion_pct",
        "spec_esperado_temporal_W",
        "nivel_spec_temporal_pct",
    ]

    base = df_temporal[cols].dropna().sort_values("time").copy()

    if base.empty:
        return pd.DataFrame(columns=["modelo", *cols])

    base["nivel_redondeado"] = base["nivel_spec_temporal_pct"].round(6)

    eventos = base.drop_duplicates(
        subset=["nivel_redondeado"],
        keep="first",
    ).copy()

    if not incluir_primer_punto and len(eventos) > 0:
        eventos = eventos.iloc[1:].copy()

    eventos.insert(0, "modelo", modelo)

    return eventos[
        [
            "modelo",
            "time",
            "utilizacion_pct",
            "spec_esperado_temporal_W",
            "nivel_spec_temporal_pct",
        ]
    ].reset_index(drop=True)


def calcular_eventos_temporales_spec(
    modelos: Dict[str, pd.DataFrame],
    spec_df: pd.DataFrame,
    variacion_muestra_spec: float,
) -> pd.DataFrame:
    eventos = []

    for modelo, df in modelos.items():
        temporal = obtener_spec_esperado_temporal(
            df,
            spec_df,
            tolerancia=variacion_muestra_spec,
        )

        ev = obtener_eventos_cambio_spec_temporal(
            temporal,
            modelo=modelo,
        )

        eventos.append(ev)

    if not eventos:
        return pd.DataFrame()

    return pd.concat(eventos, ignore_index=True)

# ---------------------------------------------------------
# Gráficas
# ---------------------------------------------------------

def anotar_eventos_sin_solape(
    ax,
    eventos: pd.DataFrame,
    max_anotaciones_eventos: int,
):
    """
    Dibuja etiquetas de puntos SPEC evitando, en lo posible, que se solapen.

    Las etiquetas muestran solo:
    - SPEC esperado
    - Utilización

    No muestran el tiempo.
    """
    if eventos.empty:
        return

    eventos = eventos.sort_values("time").reset_index(drop=True)

    if max_anotaciones_eventos is not None and max_anotaciones_eventos > 0:
        eventos = eventos.head(max_anotaciones_eventos)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    dx = x_range * 0.045
    dy = y_range * 0.075

    min_dx = x_range * 0.09
    min_dy = y_range * 0.09

    posiciones_usadas = []

    for i, row in eventos.iterrows():
        x = float(row["time"])
        y = float(row["spec_esperado_temporal_W"])

        texto = (
            f"SPEC={row['spec_esperado_temporal_W']:.1f}W\n"
            f"u={row['nivel_spec_temporal_pct']:.0f}%"
        )

        # Si el punto está a la izquierda, se intenta poner la etiqueta a la derecha.
        # Si está a la derecha, se intenta poner la etiqueta a la izquierda.
        direccion_x_preferida = 1 if x < (x_min + x_max) / 2 else -1

        candidatos = [
            (direccion_x_preferida * dx, dy),
            (direccion_x_preferida * dx, 2 * dy),
            (direccion_x_preferida * dx, -dy),
            (direccion_x_preferida * dx, 3 * dy),
            (direccion_x_preferida * dx, -2 * dy),
            (-direccion_x_preferida * dx, dy),
            (-direccion_x_preferida * dx, 2 * dy),
            (-direccion_x_preferida * dx, -dy),
            (-direccion_x_preferida * dx, 3 * dy),
            (-direccion_x_preferida * dx, -2 * dy),
            (2 * direccion_x_preferida * dx, dy),
            (2 * direccion_x_preferida * dx, 2 * dy),
            (2 * direccion_x_preferida * dx, -dy),
        ]

        mejor_tx = x + candidatos[0][0]
        mejor_ty = y + candidatos[0][1]
        mejor_score = -1

        for off_x, off_y in candidatos:
            tx = x + off_x
            ty = y + off_y

            # Evitar que la etiqueta se salga demasiado de los límites.
            tx = min(max(tx, x_min + x_range * 0.02), x_max - x_range * 0.02)
            ty = min(max(ty, y_min + y_range * 0.04), y_max - y_range * 0.04)

            if not posiciones_usadas:
                mejor_tx, mejor_ty = tx, ty
                break

            distancias = []
            for ux, uy in posiciones_usadas:
                dist_x = abs(tx - ux) / min_dx
                dist_y = abs(ty - uy) / min_dy
                distancias.append(dist_x + dist_y)

            score = min(distancias)

            if score > mejor_score:
                mejor_score = score
                mejor_tx = tx
                mejor_ty = ty

            # Si está suficientemente separado, se acepta.
            if all(
                abs(tx - ux) > min_dx or abs(ty - uy) > min_dy
                for ux, uy in posiciones_usadas
            ):
                mejor_tx = tx
                mejor_ty = ty
                break

        posiciones_usadas.append((mejor_tx, mejor_ty))

        ax.annotate(
            texto,
            xy=(x, y),
            xytext=(mejor_tx, mejor_ty),
            textcoords="data",
            fontsize=8,
            ha="left" if mejor_tx >= x else "right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="gray",
                alpha=0.82,
            ),
            arrowprops=dict(
                arrowstyle="->",
                lw=0.7,
                alpha=0.75,
            ),
        )


def crear_graficas(
    spec_df: pd.DataFrame,
    modelos: Dict[str, pd.DataFrame],
    comp: pd.DataFrame,
    metricas: pd.DataFrame,
    metas: Dict[str, Dict[str, str]],
    spec_path: str,
    tiempo_sin_arranque: float,
    multiplicador_spec_servidores: float,
    max_anotaciones_eventos: int,
    variacion_muestra_spec: float,
    salto_puntos_modelo_temporal: int = 1,
    save_fig: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("Comparación MintEDGE respecto a SPEC")
    salto_puntos_modelo_temporal = max(1, int(salto_puntos_modelo_temporal))
    # Se deja más espacio abajo para:
    # - leyenda explicativa
    # - leyenda normal
    # - botones
    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.04, right=0.98)

    indice = [0]

    texto_leyenda_fig = None

    def limpiar_leyenda_externa():
        nonlocal texto_leyenda_fig

        if texto_leyenda_fig is not None:
            texto_leyenda_fig.remove()
            texto_leyenda_fig = None

    # ---------------------------------------------------------
    # Gráfica 1
    # ---------------------------------------------------------
    def grafica_spec_vs_modelo():
        nonlocal texto_leyenda_fig

        ax.clear()
        ax.set_axis_on()
        limpiar_leyenda_externa()

        # Solo se dibujan niveles SPEC donde al menos un modelo tiene dato válido.
        util_validas = sorted(
            comp.loc[
                comp["modelo_power_W"].notna(),
                "utilizacion_pct",
            ]
            .dropna()
            .unique()
        )

        spec_plot = (
            spec_df[spec_df["utilizacion_pct"].isin(util_validas)]
            .sort_values("utilizacion_pct")
            .copy()
        )

        if spec_plot.empty:
            ax.text(
                0.5,
                0.5,
                "No hay puntos comparables dentro del rango de utilización medido.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title("Potencia del modelo energético de MintEDGE frente a SPEC")
            ax.grid(True)
            return

        spec_label = "SPEC - valor esperado"

        if multiplicador_spec_servidores != 1:
            spec_label += f" x {multiplicador_spec_servidores:g} servidores"

        ax.plot(
            spec_plot["utilizacion_pct"],
            spec_plot["spec_power_W"],
            marker="o",
            linewidth=2.5,
            label=spec_label,
        )

        for modelo, g in comp.groupby("modelo"):
            g = (
                g.dropna(subset=["modelo_power_W", "spec_power_W"])
                .sort_values("utilizacion_pct")
                .copy()
            )

            if g.empty:
                continue

            ax.plot(
                g["utilizacion_pct"],
                g["modelo_power_W"],
                marker="x",
                linewidth=2,
                label=f"{modelo} - MintEDGE",
            )

            for _, row in g.iterrows():
                ax.plot(
                    [row["utilizacion_pct"], row["utilizacion_pct"]],
                    [row["spec_power_W"], row["modelo_power_W"]],
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.45,
                    label="_nolegend_",
                )

        ax.set_xlabel("Utilización (%)")
        ax.set_ylabel("Potencia (W)")

        ax.set_title(
            "Potencia del modelo energético de MintEDGE frente a SPEC\n"
            "Punto redondo = SPEC | Cruz = MintEDGE | Línea discontinua = diferencia"
        )

        ax.set_xticks(util_validas)
        ax.grid(True)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            fontsize=9,
            frameon=True,
        )
    # ---------------------------------------------------------
    # Gráfica 2
    # ---------------------------------------------------------
    def grafica_temporal_potencia_con_spec_esperado():
        nonlocal texto_leyenda_fig

        ax.clear()
        ax.set_axis_on()
        limpiar_leyenda_externa()

        spec_label_usado = False
        punto_label_usado = False

        primer_modelo = next(iter(modelos.keys()))
        color_spec = "black"        

        for modelo, df in modelos.items():
            temporal = obtener_spec_esperado_temporal(
                df,
                spec_df,
                tolerancia=variacion_muestra_spec,
            )

            # Potencia real de MintEDGE
            temporal_mintedge_plot = (
                temporal[["time", "power_W"]]
                .dropna()
                .sort_values("time")
                .iloc[::salto_puntos_modelo_temporal]
                .copy()
            )

            if salto_puntos_modelo_temporal > 1:
                label_mintedge = (
                    f"{modelo} - potencia MintEDGE "
                    f"(1 de cada {salto_puntos_modelo_temporal} puntos)"
                )
            else:
                label_mintedge = f"{modelo} - potencia MintEDGE"

            ax.plot(
                temporal_mintedge_plot["time"],
                temporal_mintedge_plot["power_W"],
                linewidth=1.7,
                label=label_mintedge,
            )

            # Puntos SPEC exactos: solo 0, 10, 20, ..., 100
            serie_spec = (
                temporal[["time", "spec_esperado_temporal_W"]]
                .dropna()
                .sort_values("time")
                .copy()
            )

            if modelo == primer_modelo and not serie_spec.empty:
                ax.plot(
                    serie_spec["time"],
                    serie_spec["spec_esperado_temporal_W"],
                    linestyle="--",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    color=color_spec,
                    label="SPEC esperado con tolerancia",
                )

                spec_label_usado = True

            # Etiquetas de puntos SPEC.
            # Ahora incluir_primer_punto=True para que también se etiqueten
            # los primeros puntos naranjas de la izquierda.
            eventos = obtener_eventos_cambio_spec_temporal(
                temporal,
                modelo=modelo,
                incluir_primer_punto=True,
            )

            if modelo == primer_modelo and not eventos.empty:
                eventos_a_mostrar = eventos.head(max_anotaciones_eventos)

                ax.scatter(
                    eventos_a_mostrar["time"],
                    eventos_a_mostrar["spec_esperado_temporal_W"],
                    s=45,
                    marker="o",
                    color=color_spec,
                    zorder=5,
                    label="Punto SPEC etiquetado",
                )

                punto_label_usado = True

                ax.relim()
                ax.autoscale_view()

                anotar_eventos_sin_solape(
                    ax=ax,
                    eventos=eventos_a_mostrar,
                    max_anotaciones_eventos=max_anotaciones_eventos,
                )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia (W)")

        titulo = "Potencia temporal de MintEDGE frente al valor SPEC esperado"

        if multiplicador_spec_servidores != 1:
            titulo += f"\nSPEC multiplicado por {multiplicador_spec_servidores:g} servidores"

        if tiempo_sin_arranque > 0:
            titulo += f". Ignorado arranque <= {tiempo_sin_arranque}s"

        ax.set_title(titulo)
        ax.grid(True)

        # Leyenda explicativa fuera de la gráfica, abajo a la izquierda.
        texto_leyenda = (
            "Leyenda:\n"
            "• Línea MintEDGE: potencia calculada por la simulación.\n"
            "• Línea/puntos negros: valor SPEC esperado.\n"
            f"• Se pinta si la utilización está a ±{variacion_muestra_spec:g}% de un nivel SPEC.\n"
            "• Los niveles SPEC son 0%, 10%, 20%, ..., 100%.\n"
            "• En 0% no baja de 0%; en 100% no pasa de 100%."
        )

        texto_leyenda_fig = fig.text(
            0.02,
            0.035,
            texto_leyenda,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec="gray",
                alpha=0.9,
            ),
        )

        # Leyenda normal fuera de la gráfica.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            fontsize=9,
            frameon=True,
        )
    # ---------------------------------------------------------
    # Gráfica 3
    # ---------------------------------------------------------
    def grafica_utilizacion_simulacion():
        nonlocal texto_leyenda_fig

        ax.clear()
        ax.set_axis_on()
        limpiar_leyenda_externa()

        hay_datos = False

        for modelo, df in modelos.items():
            serie = (
                df[["time", "utilizacion_pct"]]
                .dropna()
                .sort_values("time")
                .drop_duplicates(subset="time", keep="last")
                .copy()
            )

            if serie.empty:
                continue

            hay_datos = True

            ax.plot(
                serie["time"],
                serie["utilizacion_pct"],
                linewidth=1.7,
                label=f"{modelo} - utilización",
            )

        if not hay_datos:
            ax.text(
                0.5,
                0.5,
                "No hay datos de utilización en la simulación.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

        titulo = "Utilización temporal de la simulación por modelo"

        if tiempo_sin_arranque > 0:
            titulo += f"\nIgnorado arranque <= {tiempo_sin_arranque}s"

        ax.set_title(titulo)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Utilización (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            fontsize=9,
            frameon=True,
        )
    
    # ---------------------------------------------------------
    # Pantalla 4: resumen numérico solo como texto
    # ---------------------------------------------------------
    def grafica_resumen_resultados():
        nonlocal texto_leyenda_fig

        ax.clear()
        limpiar_leyenda_externa()

        # Oculta completamente la gráfica/ejes.
        ax.set_axis_off()

        def formatear_valor(v, decimales=4):
            if pd.isna(v):
                return "NaN"
            return f"{v:.{decimales}f}"

        lineas = []

        lineas.append("---------------- MÉTRICAS DE ERROR ----------------")
        lineas.append("")

        for _, row in metricas.iterrows():
            lineas.append(f"{row['modelo']}:")
            lineas.append(f"  MAE: {formatear_valor(row['MAE_W'])} W")
            lineas.append(f"  RMSE: {formatear_valor(row['RMSE_W'])} W")
            lineas.append(f"  MAPE: {formatear_valor(row['MAPE_pct'])} %")
            lineas.append(f"  Sesgo medio: {formatear_valor(row['sesgo_medio_W'])} W")
            lineas.append(f"  Máximo error absoluto: {formatear_valor(row['max_error_abs_W'])} W")
            lineas.append(
                f"  Máximo error absoluto relativo: "
                f"{formatear_valor(row['max_error_abs_pct'])} %"
            )

            if "potencia_media_sobre_SPEC_pct" in row.index:
                lineas.append(
                    f"  Potencia media del modelo respecto a SPEC: "
                    f"{formatear_valor(row['potencia_media_sobre_SPEC_pct'])} %"
                )

            lineas.append("")

        lineas.append("---------------- PUNTOS COMPARADOS ----------------")

        columnas_puntos = [
            "modelo",
            "utilizacion_pct",
            "spec_power_W",
            "modelo_power_W",
            "potencia_modelo_sobre_SPEC_pct",
            "diferencia_respecto_SPEC_pct",
            "error_W",
            "error_pct",
            "n_muestras_ventana",
            "origen_punto_modelo",
        ]

        columnas_puntos = [c for c in columnas_puntos if c in comp.columns]

        tabla_puntos = comp[columnas_puntos].copy()

        with pd.option_context("display.max_rows",200,"display.max_columns",20,"display.width",220,):
            texto_tabla_puntos = tabla_puntos.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "NaN",
            )

        lineas.append(texto_tabla_puntos)
        lineas.append("")

        lineas.append("---------------- PORCENTAJE DE POTENCIA RESPECTO A SPEC ----------------")

        tabla_pct = comp.dropna(subset=["modelo_power_W"]).copy()

        columnas_pct = [
            "modelo",
            "utilizacion_pct",
            "spec_power_W",
            "modelo_power_W",
            "potencia_modelo_sobre_SPEC_pct",
            "diferencia_respecto_SPEC_pct",
        ]

        columnas_pct = [c for c in columnas_pct if c in tabla_pct.columns]

        if tabla_pct.empty:
            lineas.append("No hay puntos válidos dentro del rango medido.")
        else:
            with pd.option_context(
                "display.max_rows",
                200,
                "display.max_columns",
                20,
                "display.width",
                220,
            ):
                texto_tabla_pct = tabla_pct[columnas_pct].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "NaN",
                )

            lineas.append(texto_tabla_pct)

        texto_final = "\n".join(lineas)

        # Texto directamente sobre la ventana, no dentro de una gráfica.
        texto_leyenda_fig = fig.text(
            0.02,
            0.96,
            texto_final,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
        )
    # GRAFICAS 
    graficas = [
        grafica_spec_vs_modelo,
        grafica_temporal_potencia_con_spec_esperado,
        grafica_utilizacion_simulacion,
        grafica_resumen_resultados,
    ]

    def siguiente(event):
        indice[0] = (indice[0] + 1) % len(graficas)
        graficas[indice[0]]()
        plt.draw()

    def anterior(event):
        indice[0] = (indice[0] - 1) % len(graficas)
        graficas[indice[0]]()
        plt.draw()

    # Botones abajo a la derecha, fuera de la zona de la gráfica.
    axprev = plt.axes([0.78, 0.055, 0.075, 0.06])
    axnext = plt.axes([0.865, 0.055, 0.075, 0.06])

    bprev = Button(axprev, "←")
    bnext = Button(axnext, "→")

    bprev.on_clicked(anterior)
    bnext.on_clicked(siguiente)

    fig._mintedge_buttons = (bprev, bnext)

    graficas[0]()

    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches="tight", dpi=150)

    plt.show()

# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------
def main(
    spec_path: str,
    input_paths: List[str],
    labels: Optional[List[str]],
    power_mode: str,
    power_col: Optional[str],
    util_col: Optional[str],
    spec_util_col: Optional[str],
    spec_power_col: Optional[str],
    tiempo_sin_arranque: float,
    metodo_puntos: str,
    ventana_utilizacion: float,
    multiplicador_spec_servidores: float,
    max_anotaciones_eventos: int,
    summary_output: Optional[str],
    save_fig: Optional[str],
    variacion_muestra_spec: float,
    salto_puntos_modelo_temporal: int,
):
    if labels is None or len(labels) == 0:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in input_paths]

    if len(labels) != len(input_paths):
        raise ValueError("El número de --labels debe coincidir con el número de --inputs")

    spec_df = cargar_spec(
        spec_path,
        spec_util_col=spec_util_col,
        spec_power_col=spec_power_col,
    )

    spec_df["spec_power_W_original"] = spec_df["spec_power_W"]
    spec_df["spec_power_W"] = spec_df["spec_power_W"] * multiplicador_spec_servidores

    modelos: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, Dict[str, str]] = {}

    for path, label in zip(input_paths, labels):
        df, meta = cargar_mintedge(
            path,
            label=label,
            power_mode=power_mode,
            power_col=power_col,
            util_col=util_col,
            tiempo_sin_arranque=tiempo_sin_arranque,
        )

        modelos[label] = df
        metas[label] = meta

    comp = comparar_con_spec(
        spec_df,
        modelos,
        metodo=metodo_puntos,
        ventana=ventana_utilizacion,
    )

    metricas = calcular_metricas_error(comp)

    

    if summary_output is not None:
        comp.to_csv(summary_output, index=False)

        metricas_path = os.path.splitext(summary_output)[0] + "_metricas.csv"
        metricas.to_csv(metricas_path, index=False)

        eventos_path = os.path.splitext(summary_output)[0] + "_eventos_temporales.csv"
        eventos_temporales = calcular_eventos_temporales_spec(
            modelos,
            spec_df,
            variacion_muestra_spec=variacion_muestra_spec,
        )
        eventos_temporales.to_csv(eventos_path, index=False)

    crear_graficas(
        spec_df=spec_df,
        modelos=modelos,
        comp=comp,
        metricas=metricas,
        metas=metas,
        spec_path=spec_path,
        tiempo_sin_arranque=tiempo_sin_arranque,
        multiplicador_spec_servidores=multiplicador_spec_servidores,
        max_anotaciones_eventos=max_anotaciones_eventos,
        variacion_muestra_spec=variacion_muestra_spec,
        salto_puntos_modelo_temporal=salto_puntos_modelo_temporal,
        save_fig=save_fig,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compara la potencia generada por MintEDGE contra puntos SPECpower "
            "por utilización: 0%, 10%, ..., 100%."
        )
    )

    parser.add_argument(
        "--spec",
        required=True,
        help="Ruta del CSV con los datos SPEC del procesador/servidor",
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Uno o varios CSV generados por MintEDGE, por ejemplo lineal/frequency/powerlaw/polinomial",
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Nombres de los modelos, en el mismo orden que --inputs",
    )

    parser.add_argument(
        "--power-mode",
        choices=["servers", "total", "dynamic", "custom"],
        default="servers",
        help=(
            "Potencia de MintEDGE a comparar. "
            "servers=dynamic_W_servers+idle_W_servers; "
            "total=servers+W_links; dynamic=solo dynamic_W_servers; "
            "custom=columna indicada con --power-col. Por defecto: servers"
        ),
    )

    parser.add_argument(
        "--power-col",
        default=None,
        help="Columna de potencia si usas --power-mode custom",
    )

    parser.add_argument(
        "--util-col",
        default=None,
        help=(
            "Columna de utilización del CSV de MintEDGE. "
            "Si no se indica, usa la media de server_util_*"
        ),
    )

    parser.add_argument(
        "--spec-util-col",
        default=None,
        help=(
            "Columna de utilización/carga del CSV SPEC. "
            "Si no se indica, se detecta automáticamente"
        ),
    )

    parser.add_argument(
        "--spec-power-col",
        default=None,
        help=(
            "Columna de potencia del CSV SPEC. "
            "Si no se indica, se detecta automáticamente"
        ),
    )

    parser.add_argument(
        "--sin-arranque",
        type=float,
        default=0,
        help="Ignora filas de MintEDGE con time <= este valor. Por defecto: 0",
    )

    parser.add_argument(
        "--metodo-puntos",
        choices=["ventana", "interpolado"],
        default="ventana",
        help=(
            "Cómo obtener el punto MintEDGE para cada utilización SPEC. "
            "ventana=media cerca del nivel SPEC; interpolado=interpolación potencia-utilización"
        ),
    )

    parser.add_argument(
        "--ventana-utilizacion",
        type=float,
        default=5.0,
        help="Ventana en puntos porcentuales alrededor de cada utilización SPEC. Por defecto: ±5",
    )

    parser.add_argument(
        "--servidores-conectados",
        type=float,
        default=1.0,
        help=(
            "Número de servidores conectados que quieres comparar contra SPEC. "
            "El valor SPEC de un servidor se multiplica por este número. Por defecto: 1"
        ),
    )

    parser.add_argument(
        "--max-anotaciones-eventos",
        type=int,
        default=12,
        help="Máximo de etiquetas t/SPEC que se dibujan en la gráfica temporal. Por defecto: 12",
    )

    parser.add_argument(
        "--summary-output",
        default=None,
        help="Opcional: ruta para guardar CSV con los puntos comparados",
    )

    parser.add_argument(
        "--save-fig",
        default=None,
        help="Opcional: guarda la primera figura renderizada en esta ruta",
    )

    parser.add_argument(
        "--variacion-muestra-spec",
        type=float,
        default=0.01,
        help=(
            "Variación permitida en puntos porcentuales para pintar puntos SPEC "
            "en la gráfica temporal. Por ejemplo, 1 acepta 29%-31% alrededor de 30%. "
            "Por defecto: 0.01"
        ),
    )

    parser.add_argument(
        "--salto-puntos-modelo-temporal",
        type=int,
        default=1,
        help=(
            "Salto de puntos para pintar la potencia MintEDGE en la gráfica temporal. "
            "Por ejemplo, 5 pinta 1 de cada 5 puntos de MintEDGE. "
            "No afecta al cálculo ni al pintado de SPEC. Por defecto: 1"
        ),
    )

    args = parser.parse_args()

    main(
        spec_path=args.spec,
        input_paths=args.inputs,
        labels=args.labels,
        power_mode=args.power_mode,
        power_col=args.power_col,
        util_col=args.util_col,
        spec_util_col=args.spec_util_col,
        spec_power_col=args.spec_power_col,
        tiempo_sin_arranque=args.sin_arranque,
        metodo_puntos=args.metodo_puntos,
        ventana_utilizacion=args.ventana_utilizacion,
        multiplicador_spec_servidores=args.servidores_conectados,
        max_anotaciones_eventos=args.max_anotaciones_eventos,
        variacion_muestra_spec=args.variacion_muestra_spec,
        salto_puntos_modelo_temporal=args.salto_puntos_modelo_temporal,
        summary_output=args.summary_output,
        save_fig=args.save_fig,
    )