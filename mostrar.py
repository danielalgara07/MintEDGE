import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main(csv_path):

    # -----------------------------
    # Cargar datos
    # -----------------------------
    df = pd.read_csv(csv_path)

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
    df["energia_total"] = (
        df["dynamic_W_servers"] +
        df["idle_W_servers"] +
        df["W_links"]
    )

    df["tasa_rechazo"] = df["total_rejected"] / df["total_requests"].replace(0, np.nan)

    delay_cols_cv = [c for c in df.columns if c.startswith("delay_") and "connected_vehicles" in c]
    delay_cols_ar = [c for c in df.columns if c.startswith("delay_") and "augmented_reality" in c]
    delay_cols_va = [c for c in df.columns if c.startswith("delay_") and "video_analysis" in c]

    df["retardo_medio_cv"] = df[delay_cols_cv].mean(axis=1)
    df["retardo_medio_ar"] = df[delay_cols_ar].mean(axis=1)
    df["retardo_medio_va"] = df[delay_cols_va].mean(axis=1)

    util_cols = [c for c in df.columns if c.startswith("server_util_")]
    df["utilizacion_media_servidores"] = df[util_cols].mean(axis=1)

    # -----------------------------
    # GRÁFICAS
    # -----------------------------
    from matplotlib.widgets import Button

    fig, ax = plt.subplots()
    nombre_archivo = os.path.basename(csv_path)
    fig.canvas.manager.set_window_title(nombre_archivo)
    
    plt.subplots_adjust(bottom=0.2)

    indice = [0]

    def grafica_consumo():
        ax.clear()
        ax.plot(df["time"], df["dynamic_W_servers"], label="Consumo dinámico servidores(dynamic) W")
        ax.plot(df["time"], df["idle_W_servers"], label="Consumo en reposo servidores(idle) W")
        ax.plot(df["time"], df["W_links"], label="Consumo enlaces(w)")
        ax.plot(df["time"], df["energia_total"], label="Consumo total", linewidth=2)

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia (W)")
        ax.set_title("Evolución del consumo energético del sistema")
        ax.legend()
        ax.grid()

    def grafica_utilizacion():
        ax.clear()
        ax.plot(df["time"], df["utilizacion_media_servidores"])

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Utilización media")
        ax.set_title("Evolución de la utilización media de los servidores")
        ax.grid()

    def grafica_consumo_sin_arranque():
        ax.clear()

        df_zoom = df[df["time"] > 30]

        ax.plot(df_zoom["time"], df_zoom["dynamic_W_servers"], label="Dinámico servidores")
        ax.plot(df_zoom["time"], df_zoom["idle_W_servers"], label="Reposo servidores")
        ax.plot(df_zoom["time"], df_zoom["W_links"], label="Enlaces")
        ax.plot(df_zoom["time"], df_zoom["energia_total"], label="Total", linewidth=2)

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia (W)")
        ax.set_title("Consumo energético sin fase de arranque")
        ax.legend()
        ax.grid()

    def grafica_consumo_solo_dinamico():
        ax.clear()

        ax.plot(
            df["time"],
            df["dynamic_W_servers"],
            label="Consumo dinámico servidores",
            linewidth=2
        )

        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Potencia dinámica (W)")
        ax.set_title("Consumo dinámico de los servidores sin enlaces")
        ax.legend()
        ax.grid()
            
    graficas = [
        grafica_consumo,
        grafica_utilizacion,
        grafica_consumo_sin_arranque,
        grafica_consumo_solo_dinamico
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
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

    bprev = Button(axprev, "←")
    bnext = Button(axnext, "→")

    bprev.on_clicked(anterior)
    bnext.on_clicked(siguiente)

    graficas[0]()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al archivo results.csv o nombre que le hayas definido")
    args = parser.parse_args()
    main(args.input)