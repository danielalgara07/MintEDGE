import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


COLUMNAS_OBLIGATORIAS = {
    "model",
    "utilization_percent",
    "total_power_w",
}


def parse_float(value: str) -> float:
    """Permite escribir decimales con punto o con coma."""
    return float(str(value).replace(",", "."))


def parse_optional_alpha(value):
    """Convierte alpha a float. Si no hay alpha, devuelve None."""
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in {"", "-", "none", "nan", "sin", "no"}:
        return None

    return parse_float(text)


def format_number(value):
    """Evita mostrar como decimal los números enteros."""
    value = round(float(value), 10)

    if value.is_integer():
        return int(value)

    return value


def format_alpha(alpha):
    """Formatea alpha para mostrarlo de forma limpia."""
    if alpha is None:
        return None

    try:
        alpha = float(alpha)
    except ValueError:
        return str(alpha)

    if alpha.is_integer():
        return str(int(alpha))

    return f"{alpha:g}"


def generate_utilization_points(step: float):
    """Genera puntos de utilización desde 0 hasta 100."""
    points = []
    current = 0.0

    while current <= 100.0:
        points.append(format_number(current))
        current += step

    if points[-1] != 100:
        points.append(100)

    return points


def normalizar_nombre_modelo(nombre: str) -> str:
    nombre = str(nombre).strip().lower()

    equivalencias = {
        "lineal": "linear",
        "linear": "linear",
        "lienal": "linear",

        "interlineal": "spec",
        "spec": "spec",

        "specpowerlaw": "powerlaw",
        "spec_power_law": "powerlaw",
        "powerlaw": "powerlaw",

        "empitical": "empirical",
        "empirical": "empirical",
        "empirico": "empirical",

        "frequency": "frequency",
        "frecuencia": "frequency",
    }

    return equivalencias.get(nombre, nombre)


def buscar_columna_alpha(df):
    """Busca una columna llamada alpha o alfa."""
    for columna in df.columns:
        if columna.strip().lower() in {"alpha", "alfa"}:
            return columna

    return None


def crear_nombre_curva(nombre_medicion, modelo, alpha):
    """Crea el nombre que aparecerá en la leyenda."""
    nombre = f"{nombre_medicion} - {modelo}"
    alpha_formateado = format_alpha(alpha)

    if alpha_formateado is not None:
        nombre += f" (alpha={alpha_formateado})"

    return nombre


def cargar_csvs(rutas_csv, alphas=None, labels=None):
    dataframes = []

    if alphas is not None and len(alphas) != len(rutas_csv):
        raise ValueError(
            "Si usas --alphas, debes pasar un alpha por cada CSV. "
            "Usa '-' o 'none' para los modelos que no tengan alpha."
        )

    if labels is not None and len(labels) != len(rutas_csv):
        raise ValueError(
            "Si usas --labels, debes pasar una etiqueta por cada CSV."
        )

    for i, ruta in enumerate(rutas_csv):
        ruta = Path(ruta)

        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")

        df = pd.read_csv(
            ruta,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            skip_blank_lines=True,
        )

        df.columns = df.columns.str.strip()
        columnas_faltantes = COLUMNAS_OBLIGATORIAS - set(df.columns)

        if columnas_faltantes:
            raise ValueError(
                f"El archivo {ruta} no tiene las columnas necesarias: "
                f"{columnas_faltantes}\n"
                f"Columnas encontradas: {list(df.columns)}\n"
                f"Primeras filas leídas:\n{df.head()}"
            )

        df = df.copy()

        df["model"] = df["model"].apply(normalizar_nombre_modelo)

        df["utilization_percent"] = pd.to_numeric(
            df["utilization_percent"],
            errors="coerce",
        )

        df["total_power_w"] = pd.to_numeric(
            df["total_power_w"],
            errors="coerce",
        )

        columna_alpha = buscar_columna_alpha(df)

        if columna_alpha is not None:
            df["alpha"] = df[columna_alpha].apply(parse_optional_alpha)
        elif alphas is not None:
            df["alpha"] = parse_optional_alpha(alphas[i])
        else:
            df["alpha"] = None

        df = df.dropna(
            subset=[
                "model",
                "utilization_percent",
                "total_power_w",
            ]
        )

        df["source_file"] = ruta.name

        if labels is not None:
            df["measurement_name"] = labels[i]
        else:
            df["measurement_name"] = ruta.stem

        df["curve_label"] = df.apply(
            lambda row: crear_nombre_curva(
                nombre_medicion=row["measurement_name"],
                modelo=row["model"],
                alpha=row["alpha"],
            ),
            axis=1,
        )

        df["curve_id"] = df.apply(
            lambda row: (
                f"{i + 1}_{ruta.name}_"
                f"{row['model']}_"
                f"{format_alpha(row['alpha'])}"
            ),
            axis=1,
        )

        dataframes.append(df)

    datos = pd.concat(dataframes, ignore_index=True)

    if datos["curve_id"].nunique() < 1:
        raise ValueError(
            "No se ha podido crear ninguna curva para representar."
        )

    return datos


def preparar_curvas(datos, step=None):
    """Prepara las curvas reales o interpoladas que se van a representar."""
    curvas = []

    for curve_id, df_curva in datos.groupby("curve_id"):
        df_curva = df_curva.copy()

        nombre_curva = df_curva["curve_label"].iloc[0]
        modelo = df_curva["model"].iloc[0]
        source_file = df_curva["source_file"].iloc[0]

        alpha_values = list(df_curva["alpha"].dropna().unique())
        alpha = alpha_values[0] if alpha_values else None

        df_curva = df_curva.sort_values("utilization_percent")

        df_curva = df_curva.groupby(
            "utilization_percent",
            as_index=False,
        )["total_power_w"].mean()

        if step is None:
            datos_curva = df_curva

        else:
            puntos_utilizacion = generate_utilization_points(step)

            serie = df_curva.set_index(
                "utilization_percent"
            )["total_power_w"]

            indice_original = list(serie.index)
            indice_completo = sorted(
                set(indice_original + puntos_utilizacion)
            )

            serie = serie.reindex(indice_completo)
            serie = serie.interpolate(method="index").ffill().bfill()
            serie = serie.loc[puntos_utilizacion]

            datos_curva = pd.DataFrame(
                {
                    "utilization_percent": puntos_utilizacion,
                    "total_power_w": serie.values,
                }
            )

        curvas.append(
            {
                "id": curve_id,
                "name": nombre_curva,
                "model": modelo,
                "alpha": alpha,
                "source_file": source_file,
                "data": datos_curva,
            }
        )

    return curvas


def calcular_rango_y_curvas(curvas):
    """
    Calcula el límite superior del eje Y.

    El eje Y empieza siempre en 0.
    """
    valores = []

    for curva in curvas:
        valores.extend(
            float(value)
            for value in curva["data"]["total_power_w"].dropna()
        )

    if not valores:
        return [0, 1]

    y_max = max(valores)

    if y_max <= 0:
        return [0, 1]

    margen_superior = y_max * 0.10

    return [0, y_max + margen_superior]


def interpolar_potencia_en_puntos(df_curva, puntos_utilizacion):
    """Interpola una curva en los puntos de utilización indicados."""
    df_curva = df_curva.copy()
    df_curva = df_curva.sort_values("utilization_percent")

    serie = df_curva.set_index(
        "utilization_percent"
    )["total_power_w"]

    indice_original = list(serie.index)
    indice_completo = sorted(
        set(indice_original + puntos_utilizacion)
    )

    serie = serie.reindex(indice_completo)
    serie = serie.interpolate(method="index").ffill().bfill()

    return serie.loc[puntos_utilizacion]


def calcular_parecido_con_spec(curvas):
    """Compara todas las curvas contra una única curva SPEC."""
    curvas_spec = [
        curva
        for curva in curvas
        if curva["model"] == "spec"
    ]

    if len(curvas_spec) == 0:
        raise ValueError(
            "No se ha encontrado ningún modelo SPEC. "
            "Debe haber exactamente una curva con model=spec."
        )

    if len(curvas_spec) > 1:
        nombres_spec = [
            curva["name"]
            for curva in curvas_spec
        ]

        raise ValueError(
            "Se ha encontrado más de un modelo SPEC. "
            "Debe haber exactamente uno.\n"
            f"Curvas SPEC encontradas: {nombres_spec}"
        )

    curva_spec = curvas_spec[0]

    curvas_a_comparar = [
        curva
        for curva in curvas
        if curva["model"] != "spec"
    ]

    if len(curvas_a_comparar) == 0:
        raise ValueError(
            "Solo se ha encontrado SPEC. "
            "Debes pasar al menos otro modelo para hacer la comparación."
        )

    df_spec = curva_spec["data"].copy()
    df_spec = df_spec.sort_values("utilization_percent")

    puntos_spec = list(df_spec["utilization_percent"])

    serie_spec = df_spec.set_index(
        "utilization_percent"
    )["total_power_w"]

    resultados = []

    for curva in curvas_a_comparar:
        serie_modelo = interpolar_potencia_en_puntos(
            df_curva=curva["data"],
            puntos_utilizacion=puntos_spec,
        )

        comparacion = pd.DataFrame(
            {
                "spec_power_w": serie_spec,
                "model_power_w": serie_modelo,
            }
        )

        comparacion = comparacion.dropna()

        if comparacion.empty:
            continue

        diferencias = (
            comparacion["model_power_w"]
            - comparacion["spec_power_w"]
        )

        errores_abs = diferencias.abs()
        errores_cuadrados = diferencias ** 2

        mae = errores_abs.mean()
        rmse = math.sqrt(errores_cuadrados.mean())
        max_error = errores_abs.max()

        spec_sin_ceros = comparacion[
            "spec_power_w"
        ].replace(0, pd.NA)

        mape = (
            (errores_abs / spec_sin_ceros) * 100
        ).dropna().mean()

        resultados.append(
            {
                "curve_id": curva["id"],
                "curve_name": curva["name"],
                "model": curva["model"],
                "alpha": curva["alpha"],
                "source_file": curva["source_file"],
                "points_compared": len(comparacion),
                "mae_w": mae,
                "rmse_w": rmse,
                "max_error_w": max_error,
                "mape_percent": mape,
            }
        )

    if not resultados:
        raise ValueError(
            "No se ha podido calcular la comparación contra SPEC."
        )

    resultados = sorted(
        resultados,
        key=lambda item: item["rmse_w"],
    )

    return curva_spec, resultados


def crear_texto_conclusion(curva_spec, resultados):
    """Crea la conclusión que se muestra dentro del HTML."""
    mejor = resultados[0]

    if len(resultados) == 1:
        texto = (
            "Conclusión: solo se ha comparado un modelo contra SPEC. "
            "El modelo que más se aproxima a SPEC es "
            f"<b>{mejor['curve_name']}</b>, con un RMSE de "
            f"<b>{mejor['rmse_w']:.4f} W</b>."
        )

    else:
        texto = (
            "Conclusión: el modelo que más se aproxima a SPEC es "
            f"<b>{mejor['curve_name']}</b>, porque tiene el menor RMSE "
            f"(<b>{mejor['rmse_w']:.4f} W</b>)."
        )

    texto += (
        f"<br>Referencia SPEC: <b>{curva_spec['name']}</b>. "
        "También se muestran MAE, error máximo y MAPE "
        "para completar la comparación."
    )

    return texto


def imprimir_ranking_parecido(curva_spec, resultados):
    """Imprime por consola el ranking de parecido contra SPEC."""
    mejor = resultados[0]

    print()
    print("=" * 80)
    print("COMPARACIÓN CONTRA SPEC")
    print("=" * 80)
    print(f"Curva SPEC de referencia: {curva_spec['name']}")
    print(f"Archivo SPEC: {curva_spec['source_file']}")
    print()
    print("Métrica principal usada: RMSE")
    print("Cuanto menor sea el RMSE, más se parece el modelo al SPEC.")
    print()
    print("MODELO MÁS PARECIDO AL SPEC:")
    print(f"  {mejor['curve_name']}")
    print(f"  RMSE: {mejor['rmse_w']:.4f} W")
    print(f"  MAE: {mejor['mae_w']:.4f} W")
    print(f"  Error máximo: {mejor['max_error_w']:.4f} W")

    if pd.notna(mejor["mape_percent"]):
        print(f"  MAPE: {mejor['mape_percent']:.4f} %")

    print()
    print("Ranking completo:")

    print(
        f"{'#':>3} "
        f"{'Curva':<45} "
        f"{'RMSE W':>12} "
        f"{'MAE W':>12} "
        f"{'MAX W':>12} "
        f"{'MAPE %':>12}"
    )

    print("-" * 105)

    for i, resultado in enumerate(resultados, start=1):
        mape = resultado["mape_percent"]

        if pd.isna(mape):
            mape_text = "-"
        else:
            mape_text = f"{mape:.4f}"

        print(
            f"{i:>3} "
            f"{resultado['curve_name']:<45} "
            f"{resultado['rmse_w']:>12.4f} "
            f"{resultado['mae_w']:>12.4f} "
            f"{resultado['max_error_w']:>12.4f} "
            f"{mape_text:>12}"
        )

    print("=" * 80)
    print()


def guardar_ranking_csv(resultados, ranking_csv):
    """Guarda el ranking de parecido en un CSV."""
    if ranking_csv is None:
        return

    columnas = [
        "curve_name",
        "model",
        "alpha",
        "source_file",
        "points_compared",
        "mae_w",
        "rmse_w",
        "max_error_w",
        "mape_percent",
    ]

    df = pd.DataFrame(resultados)
    df = df[columnas]

    df.to_csv(
        ranking_csv,
        index=False,
        encoding="utf-8",
    )

    print(f"Ranking guardado en: {ranking_csv}")


def crear_tabla_ranking(resultados):
    """Crea la tabla que se muestra en la vista de comparación."""
    posiciones = []
    curvas = []
    modelos = []
    alphas = []
    puntos = []
    rmse = []
    mae = []
    max_error = []
    mape = []

    for i, resultado in enumerate(resultados, start=1):
        posiciones.append(i)
        curvas.append(resultado["curve_name"])
        modelos.append(resultado["model"])

        alpha_formateado = format_alpha(resultado["alpha"])

        if alpha_formateado is None:
            alphas.append("-")
        else:
            alphas.append(alpha_formateado)

        puntos.append(resultado["points_compared"])
        rmse.append(f"{resultado['rmse_w']:.4f}")
        mae.append(f"{resultado['mae_w']:.4f}")
        max_error.append(f"{resultado['max_error_w']:.4f}")

        if pd.isna(resultado["mape_percent"]):
            mape.append("-")
        else:
            mape.append(f"{resultado['mape_percent']:.4f}")

    tabla = go.Table(
        header=dict(
            values=[
                "Posición",
                "Curva",
                "Modelo",
                "Alpha",
                "Puntos",
                "RMSE (W)",
                "MAE (W)",
                "Error máx. (W)",
                "MAPE (%)",
            ],
            align="left",
        ),
        cells=dict(
            values=[
                posiciones,
                curvas,
                modelos,
                alphas,
                puntos,
                rmse,
                mae,
                max_error,
                mape,
            ],
            align="left",
        ),
        domain=dict(
            x=[0.02, 0.63],
            y=[0.15, 0.95],
        ),
        columnwidth=[
            50,
            180,
            60,
            40,
            45,
            70,
            65,
            100,
            70,
        ],
    )

    return tabla


def configurar_grafica_curvas(
    fig_curvas,
    curvas,
    solo_spec,
    mejor=None,
):
    """Añade las curvas y aplica el formato de la gráfica."""
    if mejor is None:
        mejor_curve_id = None
    else:
        mejor_curve_id = mejor["curve_id"]

    for curva in curvas:
        df_curva = curva["data"]

        alpha_formateado = format_alpha(curva["alpha"])

        if alpha_formateado is not None:
            hover_alpha = f"<br>Alpha: {alpha_formateado}"
        else:
            hover_alpha = ""

        nombre_traza = curva["name"]

        if curva["id"] == mejor_curve_id:
            nombre_traza = "★ " + nombre_traza

        opciones_traza = {}

        if curva["model"] == "spec":
            opciones_traza = {
                "line": dict(
                    color="#0066cc",
                    width=4,
                ),
                "marker": dict(
                    color="#0066cc",
                    size=10,
                    line=dict(
                        color="#003f7f",
                        width=2,
                    ),
                ),
            }

        fig_curvas.add_trace(
            go.Scatter(
                x=df_curva["utilization_percent"],
                y=df_curva["total_power_w"],
                mode="lines+markers",
                name=nombre_traza,
                showlegend=not solo_spec,
                hovertemplate=(
                    "Curva: "
                    + curva["name"]
                    + "<br>Modelo: "
                    + curva["model"]
                    + hover_alpha
                    + "<br>Archivo: "
                    + curva["source_file"]
                    + "<br>Utilización: %{x}%"
                    + "<br>Potencia total: %{y:.2f} W"
                    + "<extra></extra>"
                ),
                **opciones_traza,
            )
        )

    if solo_spec:
        titulo_curvas = "Potencia total del modelo SPEC"

    else:
        titulo_curvas = (
            "Comparación de potencia total por medición"
            f"<br><sup>★ Más parecido a SPEC: "
            f"{mejor['curve_name']} "
            f"(RMSE={mejor['rmse_w']:.4f} W)</sup>"
        )

    fig_curvas.update_layout(
        title=dict(
            text=titulo_curvas,
            font=dict(size=24),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(
                text="Utilización (%)",
                font=dict(size=26),
            ),
            tickfont=dict(size=20),
            range=[-1, 108],
            dtick=10,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Potencia total (W)",
                font=dict(size=26),
            ),
            tickfont=dict(size=20),
            range=calcular_rango_y_curvas(curvas),
            nticks=20,
            zeroline=False,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        showlegend=not solo_spec,
        margin=dict(
            l=110,
            r=90,
            t=95,
            b=85,
        ),
        legend=dict(
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            font=dict(size=17),
        ),
    )


def crear_html_solo_spec(div_curvas):
    """Crea un HTML que contiene únicamente la gráfica SPEC."""
    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Modelo energético SPEC</title>

    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background: white;
            overflow: hidden;
        }}

        #grafica-curvas-plotly {{
            width: 100% !important;
            height: 100% !important;
        }}

        #grafica-curvas-plotly .plot-container,
        #grafica-curvas-plotly .svg-container {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>

<body>
    {div_curvas}

    <script>
        function ajustarGrafica() {{
            const grafica = document.getElementById(
                "grafica-curvas-plotly"
            );

            if (grafica && window.Plotly) {{
                Plotly.Plots.resize(grafica);
            }}
        }}

        window.addEventListener(
            "resize",
            ajustarGrafica
        );

        setTimeout(
            ajustarGrafica,
            300
        );
    </script>
</body>
</html>
"""


def crear_html_comparacion(
    div_curvas,
    div_comparacion,
    conclusion,
    texto_leyenda,
):
    """Crea el HTML con las dos vistas."""
    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <title>Comparación de modelos energéticos</title>

    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background: white;
            overflow: hidden;
        }}

        .botonera {{
            height: 52px;
            box-sizing: border-box;
            background: white;
            padding: 8px 12px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}

        .boton {{
            padding: 8px 16px;
            margin: 0 4px;
            border: 1px solid #b8c7d9;
            border-radius: 4px;
            background: #f8fbff;
            color: #1f3b5d;
            cursor: pointer;
            font-size: 14px;
        }}

        .boton:hover {{
            background: #eaf2ff;
        }}

        .boton-activo {{
            background: #dbeafe;
            font-weight: bold;
        }}

        .vista {{
            display: none;
            width: 100%;
            height: calc(100vh - 52px);
        }}

        .vista-activa {{
            display: block;
        }}

        #vista-comparacion.vista-activa {{
            display: flex;
            flex-direction: column;
        }}

        .leyenda-metricas {{
            display: inline-block;
            margin-left: 12px;
            padding: 7px 10px;
            border: 1px solid #b8c7d9;
            border-radius: 4px;
            background: #f8fbff;
            color: #1f3b5d;
            font-size: 13px;
            cursor: help;
        }}

        .contenedor {{
            width: 100%;
            height: calc(100vh - 52px);
        }}

        .comparacion-header {{
            flex: 0 0 auto;
            padding: 14px 24px 8px 24px;
            border-bottom: 1px solid #eeeeee;
            background: white;
        }}

        .comparacion-header h2 {{
            margin: 0 0 8px 0;
            font-size: 22px;
            color: #222;
        }}

        .comparacion-header p {{
            margin: 0;
            font-size: 14px;
            line-height: 1.35;
            color: #333;
        }}

        .tabla-wrapper {{
            flex: 1 1 auto;
            min-height: 0;
            width: 100%;
        }}

        #grafica-curvas-plotly,
        #grafica-comparacion-plotly {{
            width: 100% !important;
            height: 100% !important;
        }}

        #grafica-curvas-plotly .plot-container,
        #grafica-comparacion-plotly .plot-container,
        #grafica-curvas-plotly .svg-container,
        #grafica-comparacion-plotly .svg-container {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>

<body>

    <div class="botonera">
        <button
            id="btn-curvas"
            class="boton boton-activo"
            onclick="mostrarVista('curvas')"
        >
            Curvas de potencia
        </button>

        <button
            id="btn-comparacion"
            class="boton"
            onclick="mostrarVista('comparacion')"
        >
            Comparación con SPEC
        </button>

        <span
            id="leyenda-metricas"
            class="leyenda-metricas"
            title="{texto_leyenda}"
            style="display: none;"
        >
            ℹ️ Leyenda de métricas
        </span>
    </div>

    <div class="contenedor">
        <div id="vista-curvas" class="vista vista-activa">
            {div_curvas}
        </div>

        <div id="vista-comparacion" class="vista">
            <div class="comparacion-header">
                <h2>Datos de comparación contra SPEC</h2>
                <p>{conclusion}</p>
            </div>

            <div class="tabla-wrapper">
                {div_comparacion}
            </div>
        </div>
    </div>

    <script>
        function resizePlot(plotId) {{
            const plot = document.getElementById(plotId);

            if (plot && window.Plotly) {{
                Plotly.Plots.resize(plot);
            }}
        }}

        function mostrarVista(vista) {{
            const vistaCurvas =
                document.getElementById("vista-curvas");

            const vistaComparacion =
                document.getElementById("vista-comparacion");

            const btnCurvas =
                document.getElementById("btn-curvas");

            const btnComparacion =
                document.getElementById("btn-comparacion");

            const leyendaMetricas =
                document.getElementById("leyenda-metricas");

            if (vista === "curvas") {{
                vistaCurvas.classList.add("vista-activa");
                vistaComparacion.classList.remove("vista-activa");

                btnCurvas.classList.add("boton-activo");
                btnComparacion.classList.remove("boton-activo");

                leyendaMetricas.style.display = "none";

                setTimeout(function() {{
                    resizePlot("grafica-curvas-plotly");
                }}, 100);
            }}

            if (vista === "comparacion") {{
                vistaCurvas.classList.remove("vista-activa");
                vistaComparacion.classList.add("vista-activa");

                btnCurvas.classList.remove("boton-activo");
                btnComparacion.classList.add("boton-activo");

                leyendaMetricas.style.display = "inline-block";

                setTimeout(function() {{
                    resizePlot("grafica-comparacion-plotly");
                }}, 100);
            }}
        }}

        window.addEventListener("resize", function() {{
            resizePlot("grafica-curvas-plotly");
            resizePlot("grafica-comparacion-plotly");
        }});

        setTimeout(function() {{
            resizePlot("grafica-curvas-plotly");
        }}, 300);
    </script>

</body>
</html>
"""


def crear_grafica(
    curvas,
    salida_html,
    curva_spec=None,
    resultados_parecido=None,
):
    solo_spec = (
        len(curvas) == 1
        and curvas[0]["model"] == "spec"
    )

    if solo_spec:
        mejor = None
    else:
        mejor = resultados_parecido[0]

    fig_curvas = go.Figure()

    configurar_grafica_curvas(
        fig_curvas=fig_curvas,
        curvas=curvas,
        solo_spec=solo_spec,
        mejor=mejor,
    )

    div_curvas = fig_curvas.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="grafica-curvas-plotly",
        default_width="100%",
        default_height="100%",
    )

    if solo_spec:
        html = crear_html_solo_spec(div_curvas)

    else:
        fig_comparacion = go.Figure()

        tabla_ranking = crear_tabla_ranking(
            resultados_parecido
        )

        fig_comparacion.add_trace(tabla_ranking)

        fig_comparacion.update_layout(
            template="plotly_white",
            autosize=True,
            margin=dict(
                l=30,
                r=30,
                t=20,
                b=30,
            ),
        )

        div_comparacion = fig_comparacion.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="grafica-comparacion-plotly",
            default_width="100%",
            default_height="100%",
        )

        conclusion = crear_texto_conclusion(
            curva_spec=curva_spec,
            resultados=resultados_parecido,
        )

        texto_leyenda = (
            "Puntos: número de puntos de utilización comparados "
            "contra SPEC.\n"
            "RMSE: raíz del error cuadrático medio. "
            "Penaliza más los errores grandes.\n"
            "MAE: error medio absoluto. "
            "Indica el error medio en vatios.\n"
            "Error máx.: mayor diferencia encontrada respecto a SPEC.\n"
            "MAPE: error porcentual medio respecto a SPEC.\n"
            "Conclusión: el mejor modelo es el que tiene menor RMSE."
        )

        html = crear_html_comparacion(
            div_curvas=div_curvas,
            div_comparacion=div_comparacion,
            conclusion=conclusion,
            texto_leyenda=texto_leyenda,
        )

    with open(
        salida_html,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(html)

    print(
        f"Gráfica generada correctamente: {salida_html}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara mediciones energéticas "
            "a partir de uno o más CSV."
        )
    )

    parser.add_argument(
        "csvs",
        nargs="+",
        help=(
            "Archivos CSV de entrada. "
            "Puedes pasar todos los que quieras."
        ),
    )

    parser.add_argument(
        "--output",
        default="comparacion_modelos_energia.html",
        help="Archivo HTML de salida.",
    )

    parser.add_argument(
        "--step",
        type=parse_float,
        default=None,
        help=(
            "Salto de utilización que se quiere pintar en la gráfica. "
            "Ejemplo: --step 10, --step 5, --step 2.5 o --step 2,5. "
            "Si no se pasa, se muestran todos los puntos existentes "
            "en los CSV."
        ),
    )

    parser.add_argument(
        "--alphas",
        nargs="*",
        default=None,
        help=(
            "Alpha de cada CSV, en el mismo orden en el que se pasan "
            "los archivos. Usa '-' o 'none' para los modelos que no "
            "tengan alpha. Ejemplo: --alphas none none 0.9 0.9"
        ),
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Nombre personalizado para cada CSV, en el mismo orden. "
            "Ejemplo: --labels spec linear powerlaw_09 empirical_09"
        ),
    )

    parser.add_argument(
        "--ranking-csv",
        default=None,
        help=(
            "Ruta opcional para guardar el ranking "
            "contra SPEC en CSV."
        ),
    )

    args = parser.parse_args()

    if args.step is not None:
        if args.step <= 0:
            raise ValueError(
                "--step debe ser mayor que 0."
            )

        if args.step > 100:
            raise ValueError(
                "--step debe ser menor o igual que 100."
            )

    datos = cargar_csvs(
        rutas_csv=args.csvs,
        alphas=args.alphas,
        labels=args.labels,
    )

    curvas = preparar_curvas(
        datos=datos,
        step=args.step,
    )

    if len(curvas) == 1:
        if curvas[0]["model"] != "spec":
            raise ValueError(
                "Si solo pasas una curva, debe ser el modelo SPEC. "
                "Para comparar otros modelos también debes incluir "
                "una curva SPEC."
            )

        print("Solo se ha recibido el modelo SPEC.")
        print(
            "Se generará únicamente la gráfica de potencia, "
            "sin leyenda ni comparación."
        )

        if args.ranking_csv is not None:
            print(
                "No se genera ranking CSV porque no hay "
                "otros modelos para comparar."
            )

        crear_grafica(
            curvas=curvas,
            salida_html=args.output,
        )

        return

    curva_spec, resultados_parecido = calcular_parecido_con_spec(
        curvas
    )

    imprimir_ranking_parecido(
        curva_spec=curva_spec,
        resultados=resultados_parecido,
    )

    guardar_ranking_csv(
        resultados=resultados_parecido,
        ranking_csv=args.ranking_csv,
    )

    crear_grafica(
        curvas=curvas,
        salida_html=args.output,
        curva_spec=curva_spec,
        resultados_parecido=resultados_parecido,
    )


if __name__ == "__main__":
    main()