import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.widgets import Button


# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------
def columna_o_cero(df, col):
    """Devuelve la columna si existe. Si no existe, devuelve ceros."""
    if col in df.columns:
        return df[col]
    return pd.Series(0, index=df.index)


def convertir_columnas_numericas(df):
    """Convierte a numéricas todas las columnas posibles."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def integrar_potencia(df, col):
    """
    Integra potencia en W respecto al tiempo en segundos.
    Resultado en julios: W * s = J
    """
    if col not in df.columns or len(df) < 2:
        return np.nan

    x = df["time"].to_numpy()
    y = df[col].fillna(0).to_numpy()

    return np.trapezoid(y, x)


def obtener_total_eventos(df, col):
    """
    Intenta obtener el total de eventos.
    Si la columna es acumulada, usa el último valor.
    Si no parece acumulada, suma la columna.
    """
    if col not in df.columns:
        return np.nan

    s = df[col].dropna()

    if len(s) == 0:
        return np.nan

    diffs = s.diff().dropna()

    # Si casi siempre aumenta, asumimos que es acumulada
    if len(diffs) > 0 and (diffs >= 0).mean() > 0.9:
        return s.iloc[-1]

    return s.sum()


def porcentaje_menor(valor_ganador, valor_perdedor):
    """
    Calcula cuánto menor es el ganador respecto al perdedor.
    """
    if pd.isna(valor_ganador) or pd.isna(valor_perdedor):
        return np.nan

    if valor_perdedor == 0:
        return np.nan

    return ((valor_perdedor - valor_ganador) / valor_perdedor) * 100


def comparacion_menor_es_mejor(nombre, v1, v2, label1, label2):
    """
    Compara dos valores donde menor es mejor.
    """
    if pd.isna(v1) or pd.isna(v2):
        return f"{nombre}: no disponible"

    if v1 < v2:
        pct = porcentaje_menor(v1, v2)
        return f"{nombre}: gana {label1} con un {pct:.2f}% menos"
    elif v2 < v1:
        pct = porcentaje_menor(v2, v1)
        return f"{nombre}: gana {label2} con un {pct:.2f}% menos"
    else:
        return f"{nombre}: empate"


# ---------------------------------------------------------
# Cargar y preparar CSV
# ---------------------------------------------------------
def cargar_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        raise ValueError(f"El archivo {csv_path} no tiene columna 'time'")

    if "dynamic_W_servers" not in df.columns:
        raise ValueError(f"El archivo {csv_path} no tiene columna 'dynamic_W_servers'")

    df = convertir_columnas_numericas(df)

    # Asegurar que time es numérico
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # Eliminar filas corruptas
    df = df.dropna(subset=["time"])

    # Ordenar por tiempo
    df = df.sort_values("time").reset_index(drop=True)

    # Si hay tiempos repetidos, quedarse con el último valor
    df = df.drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    # -----------------------------
    # Métricas derivadas
    # -----------------------------
    df["potencia_total"] = (
        columna_o_cero(df, "dynamic_W_servers") +
        columna_o_cero(df, "idle_W_servers") +
        columna_o_cero(df, "W_links")
    )

    if "total_requests" in df.columns and "total_rejected" in df.columns:
        df["tasa_rechazo"] = (
            df["total_rejected"] /
            df["total_requests"].replace(0, np.nan)
        ) * 100
    else:
        df["tasa_rechazo"] = np.nan

    # Retardos por servicio
    delay_cols_cv = [
        c for c in df.columns
        if c.startswith("delay_") and "connected_vehicles" in c
    ]

    delay_cols_ar = [
        c for c in df.columns
        if c.startswith("delay_") and "augmented_reality" in c
    ]

    delay_cols_va = [
        c for c in df.columns
        if c.startswith("delay_") and "video_analysis" in c
    ]

    df["retardo_medio_cv"] = df[delay_cols_cv].mean(axis=1) if delay_cols_cv else np.nan
    df["retardo_medio_ar"] = df[delay_cols_ar].mean(axis=1) if delay_cols_ar else np.nan
    df["retardo_medio_va"] = df[delay_cols_va].mean(axis=1) if delay_cols_va else np.nan

    # Utilización media de servidores
    util_cols = [c for c in df.columns if c.startswith("server_util_")]
    df["utilizacion_media_servidores"] = df[util_cols].mean(axis=1) if util_cols else np.nan

    return df


# ---------------------------------------------------------
# Calcular resumen de cada CSV
# ---------------------------------------------------------
def calcular_resumen(df):
    energia_dinamica_J = integrar_potencia(df, "dynamic_W_servers")
    energia_total_J = integrar_potencia(df, "potencia_total")

    total_requests = obtener_total_eventos(df, "total_requests")
    total_rejected = obtener_total_eventos(df, "total_rejected")

    if pd.isna(total_requests) or total_requests == 0:
        energia_dinamica_por_request = np.nan
        tasa_rechazo_global = np.nan
    else:
        energia_dinamica_por_request = energia_dinamica_J / total_requests
        tasa_rechazo_global = (total_rejected / total_requests) * 100

    resumen = {
        "potencia_dinamica_media_W": df["dynamic_W_servers"].mean(),
        "potencia_dinamica_max_W": df["dynamic_W_servers"].max(),
        "energia_dinamica_J": energia_dinamica_J,
        "energia_dinamica_Wh": energia_dinamica_J / 3600,
        "potencia_total_media_W": df["potencia_total"].mean(),
        "energia_total_J": energia_total_J,
        "energia_total_Wh": energia_total_J / 3600,
        "energia_dinamica_por_request_J": energia_dinamica_por_request,
        "total_requests": total_requests,
        "total_rejected": total_rejected,
        "tasa_rechazo_global_pct": tasa_rechazo_global,
        "retardo_medio_cv": df["retardo_medio_cv"].mean(),
        "retardo_medio_ar": df["retardo_medio_ar"].mean(),
        "retardo_medio_va": df["retardo_medio_va"].mean(),
        "utilizacion_media_servidores_pct": df["utilizacion_media_servidores"].mean() * 100,
        "usuarios_activos_medios": df["active_users"].mean() if "active_users" in df.columns else np.nan,
    }

    return resumen


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------
def main(csv_path_1, csv_path_2, label1, label2, tiempo_sin_arranque, summary_output):

    # -----------------------------
    # Cargar datos
    # -----------------------------
    df1 = cargar_csv(csv_path_1)
    df2 = cargar_csv(csv_path_2)

    resumen1 = calcular_resumen(df1)
    resumen2 = calcular_resumen(df2)

    # -----------------------------
    # Comparaciones principales
    # -----------------------------
    comparaciones = [
        comparacion_menor_es_mejor(
            "Potencia dinámica media",
            resumen1["potencia_dinamica_media_W"],
            resumen2["potencia_dinamica_media_W"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Energía dinámica total",
            resumen1["energia_dinamica_Wh"],
            resumen2["energia_dinamica_Wh"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Potencia dinámica máxima",
            resumen1["potencia_dinamica_max_W"],
            resumen2["potencia_dinamica_max_W"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Energía dinámica por request",
            resumen1["energia_dinamica_por_request_J"],
            resumen2["energia_dinamica_por_request_J"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Tasa de rechazo",
            resumen1["tasa_rechazo_global_pct"],
            resumen2["tasa_rechazo_global_pct"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Retardo medio connected vehicles",
            resumen1["retardo_medio_cv"],
            resumen2["retardo_medio_cv"],
            label1,
            label2
        ),
        comparacion_menor_es_mejor(
            "Retardo medio augmented reality",
            resumen1["retardo_medio_ar"],
            resumen2["retardo_medio_ar"],
            label1,
            label2
        ),
        
    ]

    print("\n================ COMPARATIVA ================\n")

    for c in comparaciones:
        print(c)

    print("\n---------------- RESUMEN NUMÉRICO ----------------")
    print(f"\n{label1}:")
    for k, v in resumen1.items():
        print(f"  {k}: {v:.4f}" if pd.notna(v) else f"  {k}: no disponible")

    print(f"\n{label2}:")
    for k, v in resumen2.items():
        print(f"  {k}: {v:.4f}" if pd.notna(v) else f"  {k}: no disponible")

    # Guardar resumen si se pide
    if summary_output is not None:
        resumen_df = pd.DataFrame({
            "metrica": list(resumen1.keys()),
            label1: list(resumen1.values()),
            label2: list(resumen2.values())
        })

        resumen_df["diferencia_pct_" + label2 + "_vs_" + label1] = (
            (resumen_df[label2] - resumen_df[label1]) /
            resumen_df[label1].replace(0, np.nan)
        ) * 100

        resumen_df.to_csv(summary_output, index=False)
        print(f"\nResumen guardado en: {summary_output}")

    # -----------------------------
    # GRÁFICAS
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    nombre1 = os.path.basename(csv_path_1)
    nombre2 = os.path.basename(csv_path_2)

    fig.canvas.manager.set_window_title(f"Comparativa: {nombre1} vs {nombre2}")

    plt.subplots_adjust(bottom=0.22)

    indice = [0]

    def grafica_consumo_dinamico():
        ax.clear()

        ax.plot(
            df1["time"],
            df1["dynamic_W_servers"],
            label=f"{label1} - dinámica",
            linewidth=2
        )

        ax.plot(
            df2["time"],
            df2["dynamic_W_servers"],
            label=f"{label2} - dinámica",
            linewidth=2
        )

        texto = comparacion_menor_es_mejor(
            "Menor consumo dinámico medio",
            resumen1["potencia_dinamica_media_W"],
            resumen2["potencia_dinamica_media_W"],
            label1,
            label2
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia dinámica (W)")
        ax.set_title("Comparación de potencia dinámica de servidores\n" + texto)
        ax.legend()
        ax.grid()

    def grafica_consumo_dinamico_sin_arranque():
        ax.clear()

        df1_zoom = df1[df1["time"] > tiempo_sin_arranque]
        df2_zoom = df2[df2["time"] > tiempo_sin_arranque]

        ax.plot(
            df1_zoom["time"],
            df1_zoom["dynamic_W_servers"],
            label=f"{label1} - dinámica sin arranque",
            linewidth=2
        )

        ax.plot(
            df2_zoom["time"],
            df2_zoom["dynamic_W_servers"],
            label=f"{label2} - dinámica sin arranque",
            linewidth=2
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia dinámica (W)")
        ax.set_title(f"Consumo dinámico sin fase de arranque > {tiempo_sin_arranque}s")
        ax.legend()
        ax.grid()


    def grafica_utilizacion():
        ax.clear()

        ax.plot(
            df1["time"],
            df1["utilizacion_media_servidores"] * 100,
            label=f"{label1} - utilización",
            linewidth=2
        )

        ax.plot(
            df2["time"],
            df2["utilizacion_media_servidores"] * 100,
            label=f"{label2} - utilización",
            linewidth=2
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Utilización media servidores (%)")
        ax.set_title("Comparación de utilización media de servidores")
        ax.legend()
        ax.grid()

    def grafica_tasa_rechazo():
        ax.clear()

        ax.plot(
            df1["time"],
            df1["tasa_rechazo"],
            label=f"{label1} - tasa rechazo",
            linewidth=2
        )

        ax.plot(
            df2["time"],
            df2["tasa_rechazo"],
            label=f"{label2} - tasa rechazo",
            linewidth=2
        )

        texto = comparacion_menor_es_mejor(
            "Menor tasa de rechazo global",
            resumen1["tasa_rechazo_global_pct"],
            resumen2["tasa_rechazo_global_pct"],
            label1,
            label2
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Tasa de rechazo (%)")
        ax.set_title("Comparación de tasa de rechazo\n" + texto)
        ax.legend()
        ax.grid()


    def grafica_resumen_texto():
        ax.clear()
        ax.axis("off")

        lineas = []
        lineas.append("RESUMEN DE COMPARACIÓN")
        lineas.append("")
        lineas.extend(comparaciones)
        lineas.append("")
        lineas.append("VALORES PRINCIPALES")
        lineas.append("")
        lineas.append(
            f"Potencia dinámica media:"
            f" {label1} = {resumen1['potencia_dinamica_media_W']:.4f} W |"
            f" {label2} = {resumen2['potencia_dinamica_media_W']:.4f} W"
        )
        lineas.append(
            f"Energía dinámica:"
            f" {label1} = {resumen1['energia_dinamica_Wh']:.6f} Wh |"
            f" {label2} = {resumen2['energia_dinamica_Wh']:.6f} Wh"
        )
        lineas.append(
            f"Energía dinámica por request:"
            f" {label1} = {resumen1['energia_dinamica_por_request_J']:.6f} J/req |"
            f" {label2} = {resumen2['energia_dinamica_por_request_J']:.6f} J/req"
        )
        lineas.append(
            f"Tasa de rechazo:"
            f" {label1} = {resumen1['tasa_rechazo_global_pct']:.4f}% |"
            f" {label2} = {resumen2['tasa_rechazo_global_pct']:.4f}%"
        )
        lineas.append(
            f"Utilización media:"
            f" {label1} = {resumen1['utilizacion_media_servidores_pct']:.4f}% |"
            f" {label2} = {resumen2['utilizacion_media_servidores_pct']:.4f}%"
        )

        texto = "\n".join(lineas)

        ax.text(
            0.02,
            0.98,
            texto,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            family="monospace"
        )

    graficas = [
        grafica_consumo_dinamico,
        grafica_consumo_dinamico_sin_arranque,
        grafica_utilizacion,
        grafica_tasa_rechazo,
        grafica_resumen_texto,
    ]

    def siguiente(event):
        indice[0] = (indice[0] + 1) % len(graficas)
        graficas[indice[0]]()
        plt.draw()

    def anterior(event):
        indice[0] = (indice[0] - 1) % len(graficas)
        graficas[indice[0]]()
        plt.draw()

    # Botones
    axprev = plt.axes([0.70, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

    bprev = Button(axprev, "←")
    bnext = Button(axnext, "→")

    bprev.on_clicked(anterior)
    bnext.on_clicked(siguiente)

    graficas[0]()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input1",
        required=True,
        help="Ruta del primer archivo CSV"
    )

    parser.add_argument(
        "--input2",
        required=True,
        help="Ruta del segundo archivo CSV"
    )

    parser.add_argument(
        "--label1",
        default="Archivo 1",
        help="Nombre que aparecerá para el primer CSV"
    )

    parser.add_argument(
        "--label2",
        default="Archivo 2",
        help="Nombre que aparecerá para el segundo CSV"
    )

    parser.add_argument(
        "--sin-arranque",
        type=float,
        default=30,
        help="Tiempo a partir del cual se ignora la fase de arranque"
    )

    parser.add_argument(
        "--summary-output",
        default=None,
        help="Opcional: ruta para guardar un CSV con el resumen comparativo"
    )

    args = parser.parse_args()

    main(
        args.input1,
        args.input2,
        args.label1,
        args.label2,
        args.sin_arranque,
        args.summary_output
    )