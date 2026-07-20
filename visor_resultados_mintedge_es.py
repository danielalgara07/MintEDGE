import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.widgets import Button


COLUMNAS_POTENCIA = ["dynamic_W_servers", "idle_W_servers", "W_links"]

# Tamaños grandes para presentaciones y proyectores.
plt.rcParams.update(
    {
        "font.size": 19,
        "axes.titlesize": 28,
        "axes.labelsize": 23,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 19,
    }
)


def entero_positivo(valor):
    """Acepta únicamente números enteros positivos."""
    try:
        numero = int(valor)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "El tamaño del grupo debe ser un número entero."
        ) from error

    if numero <= 0:
        raise argparse.ArgumentTypeError(
            "El tamaño del grupo debe ser mayor que cero."
        )

    return numero


def cargar_datos(ruta_csv):
    """Carga el archivo CSV y comprueba las columnas obligatorias."""
    datos = pd.read_csv(ruta_csv)

    columnas_necesarias = ["time"] + COLUMNAS_POTENCIA
    columnas_ausentes = [
        columna for columna in columnas_necesarias if columna not in datos.columns
    ]

    if columnas_ausentes:
        raise ValueError(
            "Faltan las siguientes columnas obligatorias en el CSV: "
            + ", ".join(columnas_ausentes)
        )

    if datos.empty:
        raise ValueError("El archivo CSV no contiene datos.")

    datos = datos.sort_values("time").reset_index(drop=True)

    for columna in columnas_necesarias:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    if datos[columnas_necesarias].isna().any().any():
        raise ValueError(
            "Las columnas time, dynamic_W_servers, idle_W_servers y W_links "
            "deben contener valores numéricos."
        )

    return datos


def calcular_energia_por_hora(datos):
    """Calcula la energía consumida en bloques fijos de 3600 segundos."""
    datos_horarios = datos.copy()
    datos_horarios["potencia_total_W"] = datos_horarios[COLUMNAS_POTENCIA].sum(
        axis=1
    )
    datos_horarios["hora_ejecucion"] = (
        (datos_horarios["time"] - 1) // 3600 + 1
    ).astype(int)

    resultado = datos_horarios.groupby("hora_ejecucion").agg(
        energia_Wh=("potencia_total_W", lambda valores: valores.sum() / 3600),
        potencia_media_W=("potencia_total_W", "mean"),
        potencia_minima_W=("potencia_total_W", "min"),
        potencia_maxima_W=("potencia_total_W", "max"),
    )

    resultado["cambio_porcentual"] = resultado["energia_Wh"].pct_change() * 100
    resultado["diferencia_Wh"] = resultado["energia_Wh"].diff()

    return resultado.reset_index()


def calcular_utilizacion_media(datos, tamano_grupo):
    """Calcula la utilización media de todos los servidores disponibles."""
    columnas_utilizacion = [
        columna for columna in datos.columns if columna.startswith("server_util_")
    ]

    if not columnas_utilizacion:
        raise ValueError(
            "No se han encontrado columnas de utilización de servidores. "
            "Se esperaban columnas que comenzasen por 'server_util_'."
        )

    utilizacion = datos[columnas_utilizacion].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    if tamano_grupo is None:
        return datos["time"].to_numpy(), utilizacion.to_numpy(), 1

    grupos = np.arange(len(datos)) // tamano_grupo
    tiempo_agrupado = datos.groupby(grupos)["time"].mean()
    utilizacion_agrupada = utilizacion.groupby(grupos).mean()

    return (
        tiempo_agrupado.to_numpy(),
        utilizacion_agrupada.to_numpy(),
        tamano_grupo,
    )


class NavegadorGraficas:
    """Muestra tres pantallas y permite avanzar o retroceder."""

    def __init__(self, datos_horarios, tiempo_utilizacion, utilizacion, tamano_grupo):
        self.datos_horarios = datos_horarios
        self.tiempo_utilizacion = tiempo_utilizacion
        self.utilizacion = utilizacion
        self.tamano_grupo = tamano_grupo
        self.pantalla_actual = 0

        self.figura, self.eje = plt.subplots(figsize=(16, 9))
        self.figura.canvas.manager.set_window_title(
            f"Resultados de MintEDGE"
        )
        plt.subplots_adjust(left=0.10, right=0.96, top=0.78, bottom=0.31)

        eje_anterior = plt.axes([0.33, 0.035, 0.15, 0.070])
        eje_siguiente = plt.axes([0.52, 0.035, 0.15, 0.070])

        self.boton_anterior = Button(eje_anterior, "Anterior")
        self.boton_siguiente = Button(eje_siguiente, "Siguiente")
        self.boton_anterior.label.set_fontsize(19)
        self.boton_siguiente.label.set_fontsize(19)

        self.boton_anterior.on_clicked(self.mostrar_anterior)
        self.boton_siguiente.on_clicked(self.mostrar_siguiente)

        self.indicador_pantalla = self.figura.text(
            0.94,
            0.065,
            "",
            ha="right",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

        self.dibujar_pantalla_actual()

    def mostrar_anterior(self, _evento):
        self.pantalla_actual = (self.pantalla_actual - 1) % 3
        self.dibujar_pantalla_actual()

    def mostrar_siguiente(self, _evento):
        self.pantalla_actual = (self.pantalla_actual + 1) % 3
        self.dibujar_pantalla_actual()

    def dibujar_pantalla_actual(self):
        self.eje.clear()

        if self.pantalla_actual == 0:
            self.dibujar_energia_por_hora()
        elif self.pantalla_actual == 1:
            self.dibujar_comparacion_horas()
        else:
            self.dibujar_utilizacion_media()

        self.indicador_pantalla.set_text(
            f"Gráfica {self.pantalla_actual + 1} de 3"
        )
        self.figura.canvas.draw_idle()

    def dibujar_energia_por_hora(self):
        horas = self.datos_horarios["hora_ejecucion"].to_numpy()
        energia = self.datos_horarios["energia_Wh"].to_numpy()

        columnas  = self.eje.bar(
            horas,
            energia,
            color="#4C78A8",
            edgecolor="#183B5B",
            linewidth=1.5,
            label="Energía consumida por hora",
            zorder=2,
        )

        if len(horas) == 1:
            tendencia = np.array([energia[0]])
        else:
            funcion_tendencia = np.poly1d(np.polyfit(horas, energia, 1))
            tendencia = funcion_tendencia(horas)

        # La línea y sus puntos usan colores distintos a las columnas.
        self.eje.plot(
            horas,
            tendencia,
            color="#D62728",
            marker="o",
            markerfacecolor="#FFD700",
            markeredgecolor="#7F0000",
            markeredgewidth=2,
            markersize=12,
            linewidth=4,
            label="Tendencia",
            zorder=3,
        )

        # Se reserva una banda superior para los valores de las columnas.
        valor_maximo = max(float(np.max(energia)), float(np.max(tendencia)), 1.0)
        limite_superior = valor_maximo * 1.15
        self.eje.set_ylim(0, limite_superior)

        # Distancia entre el final de la columna y el texto.
        separacion_texto = limite_superior * 0.040

        for columna, valor in zip(columnas, energia):
            centro_columna = columna.get_x() + columna.get_width() / 2
            parte_superior = columna.get_height()

            self.eje.text(
                centro_columna,
                parte_superior - separacion_texto,
                f"{valor:.2f}",
                ha="center",
                va="top",
                fontsize=13.5,
                fontweight="bold",
                color="black",
                zorder=5,
            )

        self.eje.set_title(
            "Energía consumida durante cada hora de ejecución",
            pad=72,
            fontweight="bold",
        )
        self.eje.set_xlabel("Hora de ejecución", labelpad=14)
        self.eje.set_ylabel("Energía consumida (Wh)", labelpad=14)
        self.eje.set_xticks(horas)
        self.eje.grid(axis="y", alpha=0.30, zorder=1)
        self.eje.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            framealpha=0.98,
        )

    def dibujar_comparacion_horas(self):
        informe = self.datos_horarios.copy()

        informe["hora_ejecucion"] = informe["hora_ejecucion"].map(
            lambda valor: f"Hora {valor}"
        )
        informe["energia_Wh"] = informe["energia_Wh"].map(
            lambda valor: f"{valor:.2f}"
        )
        informe["diferencia_Wh"] = informe["diferencia_Wh"].map(
            lambda valor: "-" if pd.isna(valor) else f"{valor:+.2f}"
        )
        informe["cambio_porcentual"] = informe["cambio_porcentual"].map(
            lambda valor: "-" if pd.isna(valor) else f"{valor:+.2f}%"
        )
        informe["potencia_media_W"] = informe["potencia_media_W"].map(
            lambda valor: f"{valor:.2f}"
        )
        informe["potencia_minima_W"] = informe["potencia_minima_W"].map(
            lambda valor: f"{valor:.2f}"
        )
        informe["potencia_maxima_W"] = informe["potencia_maxima_W"].map(
            lambda valor: f"{valor:.2f}"
        )

        # No se incluye ninguna columna de segundos. Cada hora siempre es 3600 s.
        columnas = [
            "hora_ejecucion",
            "energia_Wh",
            "diferencia_Wh",
            "cambio_porcentual",
            "potencia_media_W",
            "potencia_minima_W",
            "potencia_maxima_W",
        ]
        etiquetas = [
            "Hora",
            "Energía\n(Wh)",
            "Diferencia\n(Wh)",
            "Variación",
            "Potencia\nmedia (W)",
            "Potencia\nmínima (W)",
            "Potencia\nmáxima (W)",
        ]

        self.eje.axis("off")

        anchos_columnas = [
            0.11,  # Hora
            0.12,  # Energía
            0.13,  # Diferencia
            0.12,  # Variación
            0.16,  # Potencia media
            0.16,  # Potencia mínima
            0.16,  # Potencia máxima
        ]

        tabla = self.eje.table(
            cellText=informe[columnas].values,
            colLabels=etiquetas,
            cellLoc="center",
            colLoc="center",
            colWidths=anchos_columnas,

            # [posición izquierda, posición inferior, ancho, alto]
            bbox=[0.02, 0.04, 0.96, 0.86],
        )

        numero_filas = len(informe)

        # Tamaños de letra separados.
        tamano_texto = 14
        tamano_encabezado = 15

        tabla.auto_set_font_size(False)

        # Altura fija para cada tipo de fila.
        altura_encabezado = 0.12
        altura_fila_normal = 0.061

        for (fila, columna), celda in tabla.get_celld().items():

            # Centrar el contenido horizontal y verticalmente.
            celda.set_text_props(
                ha="center",
                va="center",
                multialignment="center",
            )

            # Espacio entre el texto y los bordes.
            celda.PAD = 0.08

            if fila == 0:
                # Encabezado.
                celda.set_height(altura_encabezado)

                celda.set_text_props(
                    ha="center",
                    va="center",
                    multialignment="center",
                    fontweight="bold",
                    fontsize=tamano_encabezado,
                    linespacing=1.05,
                )

            else:
                # Filas con datos.
                celda.set_height(altura_fila_normal)

                celda.set_text_props(
                    ha="center",
                    va="center",
                    fontsize=tamano_texto,
                )

                

        self.eje.set_title(
            "Comparación entre horas de ejecución\n"
            "Los valores positivos indican un aumento del consumo",
            pad=30,
            fontweight="bold",
        )

    def dibujar_utilizacion_media(self):
        self.eje.plot(
            self.tiempo_utilizacion,
            self.utilizacion * 100,
            color="#2A9D8F",
            marker="o",
            markerfacecolor="#F4A261",
            markeredgecolor="#7A3E00",
            markeredgewidth=1.2,
            markersize=6,
            linewidth=3,
        )

        if self.tamano_grupo == 1:
            texto_agrupamiento = "Sin agrupamiento: se muestran todos los puntos del CSV"
        else:
            texto_agrupamiento = (
                f"Cada punto representa la media de {self.tamano_grupo} puntos del CSV"
            )

        self.eje.set_title(
            "Utilización media de los servidores a lo largo del tiempo\n"
            + texto_agrupamiento,
            pad=20,
            fontweight="bold",
        )
        self.eje.set_xlabel("Tiempo de simulación (segundos)", labelpad=14)
        self.eje.set_ylabel("Utilización media de los servidores (%)", labelpad=14)
        self.eje.set_ylim(0, 100)

        # Mostrar aproximadamente 10 números en el eje X.
        self.eje.xaxis.set_major_locator(
            MaxNLocator(nbins=10, integer=True)
        )

        # Mostrar el eje Y de 10 en 10.
        self.eje.yaxis.set_major_locator(
            MultipleLocator(10)
        )

        # Eliminar las divisiones secundarias.
        self.eje.minorticks_off()

        # Hacer más visibles los pequeños guiones del eje X.
        self.eje.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="out",
            length=12,
            width=2.5,
            labelrotation=0,
            labelsize=16,
            pad=10,
        )

        # Configurar también las marcas del eje Y.
        self.eje.tick_params(
            axis="y",
            which="major",
            left=True,
            right=False,
            direction="out",
            length=8,
            width=2,
            labelsize=16,
            pad=8,
        )

        # Mostrar únicamente la cuadrícula principal.
        self.eje.grid(
            True,
            which="major",
            axis="both",
            linewidth=1,
            alpha=0.30,
        )

    def mostrar(self):
        plt.show()


def main():
    analizador = argparse.ArgumentParser(
        description="Muestra los resultados de energía y utilización de MintEDGE."
    )
    analizador.add_argument(
        "archivo_csv",
        help="Ruta del archivo CSV generado por MintEDGE",
    )
    analizador.add_argument(
        "tamano_grupo",
        nargs="?",
        type=entero_positivo,
        default=None,
        help=(
            "Número opcional de puntos del CSV usados para calcular cada media "
            "de utilización. Solo se aceptan enteros positivos."
        ),
    )
    argumentos = analizador.parse_args()

    datos = cargar_datos(argumentos.archivo_csv)
    datos_horarios = calcular_energia_por_hora(datos)
    tiempo_utilizacion, utilizacion, tamano_utilizado = calcular_utilizacion_media(
        datos,
        argumentos.tamano_grupo,
    )

    navegador = NavegadorGraficas(
        datos_horarios,
        tiempo_utilizacion,
        utilizacion,
        tamano_utilizado,
    )
    navegador.mostrar()


if __name__ == "__main__":
    main()
