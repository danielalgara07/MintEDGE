import argparse
import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


COLUMNAS_POTENCIA = ["dynamic_W_servers", "idle_W_servers"]


def leer_modelo(texto):
    """Convierte nombre=ruta.csv en una pareja con el nombre y la ruta."""
    if "=" not in texto:
        raise argparse.ArgumentTypeError(
            "Cada modelo debe escribirse como nombre=ruta.csv"
        )

    nombre, ruta = texto.split("=", 1)
    nombre = nombre.strip()
    ruta = ruta.strip()

    if not nombre or not ruta:
        raise argparse.ArgumentTypeError(
            "Cada modelo debe escribirse como nombre=ruta.csv"
        )

    return nombre, Path(ruta)


def cargar_csv(nombre, ruta, intervalo):
    """Carga un CSV y calcula la energía y la utilización por intervalos."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el archivo del modelo {nombre}: {ruta}"
        )

    datos = pd.read_csv(ruta)

    obligatorias = ["time"] + COLUMNAS_POTENCIA
    faltan = [columna for columna in obligatorias if columna not in datos.columns]

    if faltan:
        raise ValueError(
            f"En {ruta} faltan estas columnas: {', '.join(faltan)}"
        )

    datos = datos.copy()

    datos["time"] = pd.to_numeric(datos["time"], errors="coerce")

    for columna in COLUMNAS_POTENCIA:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

    datos = datos.dropna(subset=obligatorias)
    datos = datos.sort_values("time")

    if datos.empty:
        raise ValueError(
            f"El archivo {ruta} no contiene datos válidos"
        )

    # Potencia total de los servidores en cada instante.
    datos["potencia_W"] = (
        datos["dynamic_W_servers"]
        + datos["idle_W_servers"]
    )

    # Se calcula cuánto tiempo representa cada fila.
    # Normalmente, cada fila de MintEDGE representa un segundo.
    diferencias = datos["time"].diff()
    diferencias_validas = diferencias[diferencias > 0]

    if diferencias_validas.empty:
        paso_normal = 1
    else:
        paso_normal = diferencias_validas.median()

    datos["duracion_s"] = diferencias.fillna(paso_normal)

    datos.loc[
        datos["duracion_s"] <= 0,
        "duracion_s"
    ] = paso_normal

    # Un vatio durante un segundo equivale a un julio.
    datos["energia_J"] = (
        datos["potencia_W"]
        * datos["duracion_s"]
    )

    # Por ejemplo, con un intervalo de 3600 segundos:
    # los segundos 1...3600 pertenecen a la hora 1,
    # los segundos 3601...7200 pertenecen a la hora 2.
    datos["bloque_s"] = (
        np.ceil(datos["time"] / intervalo)
        * intervalo
    )

    energia = (
        datos.groupby("bloque_s", as_index=False)["energia_J"]
        .sum()
        .rename(columns={"bloque_s": "time"})
    )

    energia["modelo"] = nombre

    # Busca todas las columnas de utilización de servidores.
    columnas_utilizacion = [
        columna
        for columna in datos.columns
        if columna.startswith("server_util_")
    ]

    utilizacion = None

    if columnas_utilizacion:
        util = datos[columnas_utilizacion].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Media de utilización de todos los servidores.
        datos["utilizacion"] = util.mean(axis=1)

        # Si la utilización está entre 0 y 1, se convierte a porcentaje.
        if datos["utilizacion"].max() <= 1.5:
            datos["utilizacion"] = (
                datos["utilizacion"]
                * 100
            )

        # Media de utilización dentro de cada intervalo.
        utilizacion = (
            datos.groupby(
                "bloque_s",
                as_index=False,
            )["utilizacion"]
            .mean()
            .rename(columns={"bloque_s": "time"})
        )

    return energia, utilizacion


def unidad_tiempo(valores, intervalo):
    """Decide si el tiempo se muestra en segundos, minutos u horas."""
    maximo = max(valores) if valores else 0

    if intervalo >= 3600 and maximo >= 3600:
        return 3600, "Tiempo de simulación (h)"

    if intervalo >= 60 and maximo >= 60:
        return 60, "Tiempo de simulación (min)"

    return 1, "Tiempo de simulación (s)"


def crear_tabla(energias, nombre_spec, divisor_tiempo):
    """Crea la tabla de diferencias respecto al modelo SPEC."""
    pivote = energias.pivot_table(
        index="time",
        columns="modelo",
        values="energia_J",
        aggfunc="sum",
    )

    if nombre_spec not in pivote.columns:
        raise ValueError(
            "No se han encontrado datos para el modelo SPEC"
        )

    filas = []

    for modelo in pivote.columns:
        if modelo == nombre_spec:
            continue

        comparacion = pivote[
            [nombre_spec, modelo]
        ].dropna()

        for tiempo, fila in comparacion.iterrows():
            energia_spec = fila[nombre_spec]
            energia_modelo = fila[modelo]

            diferencia = (
                energia_modelo
                - energia_spec
            )

            if energia_spec != 0:
                diferencia_porcentaje = (
                    diferencia
                    / energia_spec
                    * 100
                )
            else:
                diferencia_porcentaje = np.nan

            filas.append(
                {
                    "Tiempo": tiempo / divisor_tiempo,
                    "Modelo": modelo,
                    "Energía modelo (J)": energia_modelo,
                    "Energía SPEC (J)": energia_spec,
                    "Diferencia (J)": diferencia,
                    "Diferencia (%)": diferencia_porcentaje,
                }
            )

    tabla = pd.DataFrame(filas)

    if not tabla.empty:
        tabla = tabla.sort_values(
            ["Tiempo", "Modelo"]
        )

    return tabla


def crear_estilo_eje_x(titulo, intervalo, divisor):
    """Crea el estilo común del eje temporal."""
    # Con intervalo 3600 y divisor 3600, el paso será 1 hora.
    paso_marcas = intervalo / divisor

    return dict(
        title=dict(
            text=titulo,
            font=dict(size=24),
            standoff=15,
        ),
        tickfont=dict(size=18),
        tickmode="linear",
        tick0=0,
        dtick=paso_marcas,
        showgrid=True,
        gridcolor="#d9d9d9",
        zeroline=False,
        rangemode="tozero",
    )


def crear_figura_energia(energias, intervalo):
    """Crea la primera vista con la energía de todos los modelos."""
    figura = go.Figure()

    modelos = list(
        energias["modelo"].drop_duplicates()
    )

    divisor, titulo_eje_x = unidad_tiempo(
        energias["time"].tolist(),
        intervalo,
    )

    for modelo in modelos:
        datos_modelo = energias[
            energias["modelo"] == modelo
        ].sort_values("time")

        figura.add_trace(
            go.Scatter(
                x=datos_modelo["time"] / divisor,
                y=datos_modelo["energia_J"],
                mode="lines+markers",
                name=modelo,
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate=(
                    f"Modelo: {modelo}<br>"
                    f"{titulo_eje_x}: %{{x}}<br>"
                    "Energía: %{y:,.2f} J"
                    "<extra></extra>"
                ),
            )
        )

    figura.update_layout(
        title=dict(
            text="Comparación de energía entre modelos",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=30),
        ),
        xaxis=crear_estilo_eje_x(
            titulo_eje_x,
            intervalo,
            divisor,
        ),
        yaxis=dict(
            title=dict(
                text="Energía consumida en el intervalo (J)",
                font=dict(size=24),
                standoff=20,
            ),
            tickfont=dict(size=18),
            showgrid=True,
            gridcolor="#d9d9d9",
            zeroline=False,
            rangemode="tozero",
        ),
        legend=dict(
            font=dict(size=19),
            orientation="h",
            x=0,
            xanchor="left",
            y=1.04,
            yanchor="bottom",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=18),
        margin=dict(
            l=120,
            r=60,
            t=170,
            b=100,
        ),
        height=760,
        hovermode="x unified",
    )

    return figura, divisor, titulo_eje_x


def crear_figura_utilizacion(
    utilizacion_spec,
    nombre_spec,
    intervalo,
    divisor,
    titulo_eje_x,
):
    """Crea la segunda vista con la utilización del modelo SPEC."""
    figura = go.Figure()

    utilizacion_spec = utilizacion_spec.sort_values(
        "time"
    )

    figura.add_trace(
        go.Scatter(
            x=utilizacion_spec["time"] / divisor,
            y=utilizacion_spec["utilizacion"],
            mode="lines+markers",
            name=f"Utilización {nombre_spec}",
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate=(
                f"{titulo_eje_x}: %{{x}}<br>"
                "Utilización media: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    figura.update_layout(
        title=dict(
            text="Utilización media del modelo SPEC",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=30),
        ),
        xaxis=crear_estilo_eje_x(
            titulo_eje_x,
            intervalo,
            divisor,
        ),
        yaxis=dict(
            title=dict(
                text="Utilización media (%)",
                font=dict(size=24),
                standoff=20,
            ),
            tickfont=dict(size=18),
            range=[0, 100],
            dtick=10,
            showgrid=True,
            gridcolor="#d9d9d9",
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=19),
            orientation="h",
            x=0,
            xanchor="left",
            y=1.04,
            yanchor="bottom",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=18),
        margin=dict(
            l=120,
            r=60,
            t=170,
            b=100,
        ),
        height=760,
        hovermode="x unified",
    )

    return figura


def crear_figura_tabla(
    energias,
    nombre_spec,
    divisor_tiempo,
):
    """Crea la tercera vista con la tabla de diferencias."""
    tabla = crear_tabla(
        energias,
        nombre_spec,
        divisor_tiempo,
    )

    if tabla.empty:
        raise ValueError(
            "No existen datos suficientes para crear la tabla de diferencias"
        )

    valores_tabla = [
        tabla["Tiempo"]
        .map(lambda valor: f"{valor:.2f}")
        .tolist(),

        tabla["Modelo"].tolist(),

        tabla["Energía modelo (J)"]
        .map(lambda valor: f"{valor:,.2f}")
        .tolist(),

        tabla["Energía SPEC (J)"]
        .map(lambda valor: f"{valor:,.2f}")
        .tolist(),

        tabla["Diferencia (J)"]
        .map(lambda valor: f"{valor:,.2f}")
        .tolist(),

        tabla["Diferencia (%)"]
        .map(
            lambda valor: (
                "-"
                if pd.isna(valor)
                else f"{valor:.2f}%"
            )
        )
        .tolist(),
    ]

    figura = go.Figure(
        data=[
            go.Table(
                columnwidth=[
                    100,
                    130,
                    180,
                    180,
                    160,
                    150,
                ],
                header=dict(
                    values=[
                        "<b>Tiempo</b>",
                        "<b>Modelo</b>",
                        "<b>Energía modelo (J)</b>",
                        "<b>Energía SPEC (J)</b>",
                        "<b>Diferencia (J)</b>",
                        "<b>Diferencia (%)</b>",
                    ],
                    align="center",
                    font=dict(size=17),
                    height=40,
                    fill_color="#eeeeee",
                    line_color="#bdbdbd",
                ),
                cells=dict(
                    values=valores_tabla,
                    align="center",
                    font=dict(size=15),
                    height=32,
                    fill_color="white",
                    line_color="#d9d9d9",
                ),
            )
        ]
    )

    altura_tabla = max(
        650,
        180 + len(tabla) * 32,
    )

    figura.update_layout(
        title=dict(
            text="Diferencias de energía respecto a SPEC",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=30),
        ),
        paper_bgcolor="white",
        font=dict(size=18),
        margin=dict(
            l=50,
            r=50,
            t=110,
            b=50,
        ),
        height=altura_tabla,
    )

    return figura


def crear_html(
    figura_energia,
    figura_utilizacion,
    figura_tabla,
    salida,
):
    """Crea un HTML con botones y tres vistas independientes."""

    configuracion = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": False,
    }

    # Plotly se incluye solamente en la primera gráfica.
    html_energia = figura_energia.to_html(
        full_html=False,
        include_plotlyjs=True,
        config=configuracion,
    )

    html_utilizacion = figura_utilizacion.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=configuracion,
    )

    html_tabla = figura_tabla.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=configuracion,
    )

    contenido = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Comparación de modelos</title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            background-color: white;
            font-family: Arial, Helvetica, sans-serif;
        }}

        .contenedor-principal {{
            width: 100%;
            max-width: 1600px;
            margin: 0 auto;
            padding: 25px 35px 40px 35px;
        }}

        .menu-vistas {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;

            margin-top: 10px;
            margin-bottom: 30px;
            padding: 15px;

            background-color: white;
            border-bottom: 1px solid #dddddd;
        }}

        .boton-vista {{
            min-width: 190px;
            padding: 13px 22px;

            border: 1px solid #888888;
            border-radius: 5px;

            background-color: white;
            color: #222222;

            font-size: 17px;
            font-weight: bold;

            cursor: pointer;
            transition: background-color 0.2s;
        }}

        .boton-vista:hover {{
            background-color: #eeeeee;
        }}

        .boton-vista.activo {{
            background-color: #333333;
            color: white;
            border-color: #333333;
        }}

        .vista {{
            display: none;
            width: 100%;
            background-color: white;
        }}

        .vista.activa {{
            display: block;
        }}

        .plotly-graph-div {{
            width: 100% !important;
        }}

        @media screen and (max-width: 700px) {{
            .contenedor-principal {{
                padding: 15px;
            }}

            .menu-vistas {{
                gap: 10px;
                margin-bottom: 20px;
            }}

            .boton-vista {{
                width: 100%;
                min-width: 0;
            }}
        }}
    </style>
</head>

<body>
    <div class="contenedor-principal">

        <div class="menu-vistas">
            <button
                id="boton-energia"
                class="boton-vista activo"
                onclick="mostrarVista('energia')"
            >
                Energía
            </button>

            <button
                id="boton-utilizacion"
                class="boton-vista"
                onclick="mostrarVista('utilizacion')"
            >
                Utilización SPEC
            </button>

            <button
                id="boton-tabla"
                class="boton-vista"
                onclick="mostrarVista('tabla')"
            >
                Tabla de diferencias
            </button>
        </div>

        <div
            id="vista-energia"
            class="vista activa"
        >
            {html_energia}
        </div>

        <div
            id="vista-utilizacion"
            class="vista"
        >
            {html_utilizacion}
        </div>

        <div
            id="vista-tabla"
            class="vista"
        >
            {html_tabla}
        </div>

    </div>

    <script>
        function mostrarVista(nombreVista) {{
            const vistas = document.querySelectorAll(".vista");
            const botones = document.querySelectorAll(".boton-vista");

            // Oculta completamente todas las vistas.
            vistas.forEach(function(vista) {{
                vista.classList.remove("activa");
                vista.style.display = "none";
            }});

            // Desactiva todos los botones.
            botones.forEach(function(boton) {{
                boton.classList.remove("activo");
            }});

            const vistaSeleccionada = document.getElementById(
                "vista-" + nombreVista
            );

            const botonSeleccionado = document.getElementById(
                "boton-" + nombreVista
            );

            // Muestra solamente la vista seleccionada.
            vistaSeleccionada.style.display = "block";
            vistaSeleccionada.classList.add("activa");
            botonSeleccionado.classList.add("activo");

            // Recalcula el tamaño de la gráfica que acaba de mostrarse.
            // Esto evita errores producidos porque estaba oculta.
            window.requestAnimationFrame(function() {{
                const graficas = vistaSeleccionada.querySelectorAll(
                    ".plotly-graph-div"
                );

                graficas.forEach(function(grafica) {{
                    if (window.Plotly) {{
                        Plotly.Plots.resize(grafica);
                    }}
                }});
            }});
        }}

        window.addEventListener("resize", function() {{
            const vistaActiva = document.querySelector(
                ".vista.activa"
            );

            if (!vistaActiva) {{
                return;
            }}

            const graficas = vistaActiva.querySelectorAll(
                ".plotly-graph-div"
            );

            graficas.forEach(function(grafica) {{
                if (window.Plotly) {{
                    Plotly.Plots.resize(grafica);
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    salida.write_text(
        contenido,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara la energía de varios modelos de MintEDGE"
        )
    )

    parser.add_argument(
        "--modelo",
        action="append",
        type=leer_modelo,
        required=True,
        help=(
            "Modelo y CSV con el formato nombre=ruta.csv. "
            "Se repite una vez por modelo."
        ),
    )

    parser.add_argument(
        "--intervalo",
        type=int,
        required=True,
        help=(
            "Segundos que se agrupan en cada punto. "
            "Por ejemplo, 3600 para una hora."
        ),
    )

    parser.add_argument(
        "--salida",
        default="comparacion_modelos.html",
        help="Nombre del archivo HTML de salida.",
    )

    parser.add_argument(
        "--no-abrir",
        action="store_true",
        help="Crea el HTML sin abrirlo en el navegador.",
    )

    args = parser.parse_args()

    if len(args.modelo) < 2:
        parser.error(
            "Debes indicar como mínimo dos modelos"
        )

    if args.intervalo <= 0:
        parser.error(
            "El intervalo debe ser mayor que cero"
        )

    nombres_minuscula = [
        nombre.lower()
        for nombre, _ in args.modelo
    ]

    if len(nombres_minuscula) != len(set(nombres_minuscula)):
        parser.error(
            "No se puede repetir el nombre de un modelo"
        )

    posicion_spec = next(
        (
            indice
            for indice, nombre
            in enumerate(nombres_minuscula)
            if nombre == "spec"
        ),
        None,
    )

    if posicion_spec is None:
        parser.error(
            'Uno de los modelos debe llamarse exactamente "spec"'
        )

    nombre_spec = args.modelo[posicion_spec][0]

    energias = []
    utilizacion_spec = None

    for nombre, ruta in args.modelo:
        energia, utilizacion = cargar_csv(
            nombre,
            ruta,
            args.intervalo,
        )

        energias.append(energia)

        if nombre.lower() == "spec":
            utilizacion_spec = utilizacion

    if utilizacion_spec is None:
        raise ValueError(
            "El CSV de SPEC no contiene columnas que "
            "empiecen por server_util_"
        )

    todas_energias = pd.concat(
        energias,
        ignore_index=True,
    )

    figura_energia, divisor, titulo_eje_x = crear_figura_energia(
        todas_energias,
        args.intervalo,
    )

    figura_utilizacion = crear_figura_utilizacion(
        utilizacion_spec,
        nombre_spec,
        args.intervalo,
        divisor,
        titulo_eje_x,
    )

    figura_tabla = crear_figura_tabla(
        todas_energias,
        nombre_spec,
        divisor,
    )

    salida = Path(args.salida)

    crear_html(
        figura_energia,
        figura_utilizacion,
        figura_tabla,
        salida,
    )

    print(
        f"Comparación creada: {salida.resolve()}"
    )

    if not args.no_abrir:
        webbrowser.open(
            "file://" + os.path.abspath(salida)
        )


if __name__ == "__main__":
    main()