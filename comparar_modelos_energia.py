import argparse
import itertools
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


COLUMNAS_OBLIGATORIAS = {
    "model",
    "utilization_percent",
    "total_power_w",
}


def normalizar_nombre_modelo(nombre: str) -> str:
    nombre = str(nombre).strip().lower()

    equivalencias = {
        "lineal": "linear",
        "linear": "linear",
        "specpowerlaw": "specpowerlaw",
        "spec_power_law": "specpowerlaw",
        "powerlaw": "specpowerlaw",
        "empitical": "empirical",  # por si está escrito así
        "empirical": "empirical",
        "empirico": "empirical",
    }

    return equivalencias.get(nombre, nombre)


def cargar_csvs(rutas_csv):
    dataframes = []

    for ruta in rutas_csv:
        ruta = Path(ruta)

        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")

        df = pd.read_csv(ruta)

        columnas_faltantes = COLUMNAS_OBLIGATORIAS - set(df.columns)
        if columnas_faltantes:
            raise ValueError(
                f"El archivo {ruta} no tiene las columnas necesarias: "
                f"{columnas_faltantes}"
            )

        df = df.copy()
        df["model"] = df["model"].apply(normalizar_nombre_modelo)
        df["utilization_percent"] = pd.to_numeric(
            df["utilization_percent"], errors="coerce"
        )
        df["total_power_w"] = pd.to_numeric(
            df["total_power_w"], errors="coerce"
        )

        df = df.dropna(subset=["model", "utilization_percent", "total_power_w"])
        df["source_file"] = ruta.name

        dataframes.append(df)

    datos = pd.concat(dataframes, ignore_index=True)

    if datos["model"].nunique() < 2:
        raise ValueError("Debes pasar al menos dos modelos diferentes.")

    return datos


def preparar_tabla_potencia(datos):
    """
    Crea una tabla donde:
    - el índice es la utilización
    - cada columna es un modelo
    - cada celda es total_power_w
    """
    tabla = datos.pivot_table(
        index="utilization_percent",
        columns="model",
        values="total_power_w",
        aggfunc="mean",
    )

    tabla = tabla.sort_index()

    # Si algún modelo no tiene exactamente los mismos puntos de utilización,
    # se interpolan los valores intermedios.
    tabla = tabla.interpolate(method="index").ffill().bfill()

    return tabla


def crear_diferencias(tabla_potencia):
    """
    Calcula diferencias entre cada par de modelos.

    Ejemplo:
    empirical - linear
    specpowerlaw - linear
    specpowerlaw - empirical
    """
    diferencias = pd.DataFrame(index=tabla_potencia.index)

    modelos = list(tabla_potencia.columns)

    for modelo_a, modelo_b in itertools.combinations(modelos, 2):
        nombre_columna = f"{modelo_b} - {modelo_a}"
        diferencias[nombre_columna] = (
            tabla_potencia[modelo_b] - tabla_potencia[modelo_a]
        )

    return diferencias


def crear_grafica(tabla_potencia, tabla_diferencias, salida_html):
    fig = go.Figure()

    trazas_potencia = []
    trazas_diferencias = []

    # Gráfica 1: utilización frente a potencia total
    for modelo in tabla_potencia.columns:
        fig.add_trace(
            go.Scatter(
                x=tabla_potencia.index,
                y=tabla_potencia[modelo],
                mode="lines+markers",
                name=modelo,
                visible=True,
                hovertemplate=(
                    "Modelo: " + modelo +
                    "<br>Utilización: %{x}%"
                    "<br>Potencia total: %{y:.2f} W"
                    "<extra></extra>"
                ),
            )
        )
        trazas_potencia.append(True)
        trazas_diferencias.append(False)

    # Gráfica 2: diferencias entre modelos
    for diferencia in tabla_diferencias.columns:
        fig.add_trace(
            go.Scatter(
                x=tabla_diferencias.index,
                y=tabla_diferencias[diferencia],
                mode="lines+markers",
                name=diferencia,
                visible=False,
                hovertemplate=(
                    "Diferencia: " + diferencia +
                    "<br>Utilización: %{x}%"
                    "<br>Diferencia: %{y:.2f} W"
                    "<extra></extra>"
                ),
            )
        )
        trazas_potencia.append(False)
        trazas_diferencias.append(True)

    fig.update_layout(
        title="Comparación de potencia total por modelo",
        xaxis_title="Utilización (%)",
        yaxis_title="Potencia total (W)",
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "y": 1.18,
                "xanchor": "center",
                "buttons": [
                    {
                        "label": "Potencia total",
                        "method": "update",
                        "args": [
                            {"visible": trazas_potencia},
                            {
                                "title": "Comparación de potencia total por modelo",
                                "yaxis": {"title": "Potencia total (W)"},
                            },
                        ],
                    },
                    {
                        "label": "Diferencias entre modelos",
                        "method": "update",
                        "args": [
                            {"visible": trazas_diferencias},
                            {
                                "title": "Diferencias de potencia entre modelos",
                                "yaxis": {"title": "Diferencia de potencia (W)"},
                            },
                        ],
                    },
                ],
            }
        ],
    )

    fig.write_html(salida_html)
    print(f"Gráfica generada correctamente: {salida_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara modelos energéticos a partir de dos o más CSV."
    )

    parser.add_argument(
        "csvs",
        nargs="+",
        help="Archivos CSV de entrada.",
    )

    parser.add_argument(
        "--output",
        default="comparacion_modelos_energia.html",
        help="Archivo HTML de salida.",
    )

    parser.add_argument(
        "--diff-csv",
        default="diferencias_modelos.csv",
        help="CSV de salida con las diferencias calculadas.",
    )

    args = parser.parse_args()

    if len(args.csvs) < 2:
        raise ValueError("Debes pasar dos o más archivos CSV.")

    datos = cargar_csvs(args.csvs)
    tabla_potencia = preparar_tabla_potencia(datos)
    tabla_diferencias = crear_diferencias(tabla_potencia)

    tabla_diferencias.to_csv(args.diff_csv)
    print(f"CSV de diferencias generado: {args.diff_csv}")

    crear_grafica(tabla_potencia, tabla_diferencias, args.output)


if __name__ == "__main__":
    main()