import argparse
import csv
import importlib.util
import sys
from pathlib import Path


# -------------------------------------------------------------------------
# Configuración fija
# -------------------------------------------------------------------------

UTILIZATION_POINTS = range(0, 101, 10)

CSV_COLUMNS = [
    "model",
    "utilization_percent",
    "dynamic_power_w",
    "idle_power_w",
    "total_power_w",
]


# -------------------------------------------------------------------------
# Carga directa de mintedge/energy.py sin ejecutar mintedge/__init__.py
# Así evitamos que importe users.py y libsumo.
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ENERGY_PATH = PROJECT_ROOT / "mintedge" / "energy.py"

if not ENERGY_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {ENERGY_PATH}")

spec = importlib.util.spec_from_file_location("energy_module", ENERGY_PATH)

if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el módulo desde: {ENERGY_PATH}")

energy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy_module)

EnergyModelServer = energy_module.EnergyModelServer
EnergyModelServerPowerLaw = energy_module.EnergyModelServerPowerLaw
EnergyModelServerEmpirical = energy_module.EnergyModelServerEmpirical
EnergyModelServerFrequency = energy_module.EnergyModelServerFrequency
EnergyModelServerSpecLinearInterpolation = (
    energy_module.EnergyModelServerSpecLinearInterpolation
)


# -------------------------------------------------------------------------
# Servidor falso para probar los modelos de energía
# -------------------------------------------------------------------------

class FakeEnv:
    def __init__(self, now=1000):
        self.now = now


class FakeServer:
    """
    Servidor mínimo para probar los modelos energéticos.

    min_power se usa como idle_power.
    max_power se usa como potencia máxima del servidor.
    """

    def __init__(
        self,
        max_power=250.0,
        min_power=100.0,
        max_cap=100.0,
        spec_power_curve=None,
    ):
        self.env = FakeEnv(now=1000)

        self.is_on = True
        self.last_onoff_time = 0
        self.boot_time = 0

        self.max_power = float(max_power)
        self.idle_power = float(min_power)

        self.max_cap = float(max_cap)
        self.used_ops = 0.0

        # Curva SPEC. En el modelo SPEC esta curva representa potencia total.
        self.spec_power_curve = spec_power_curve

        # Parámetros usados por EnergyModelServerFrequency.
        #
        # Se calibran para que:
        #   - al 0% la potencia total sea aproximadamente min_power
        #   - al 100% la potencia total sea aproximadamente max_power
        #
        # Como el modelo frequency calcula:
        #   dynamic = activity_factor * capacitance * voltage^2 * frequency
        #
        # elegimos activity_factor automáticamente para que dynamic al 100%
        # sea max_power - min_power.
        self.capacitance = 45e-9

        self.min_frequency = 0.0
        self.max_frequency = 3.0e9

        self.min_voltage = 0.75
        self.max_voltage = 1.10

        dynamic_target = max(self.max_power - self.idle_power, 0.0)
        denominator = self.capacitance * (self.max_voltage ** 2) * self.max_frequency

        if denominator > 0:
            self.activity_factor = dynamic_target / denominator
        else:
            self.activity_factor = 0.0

    def set_utilization_percent(self, utilization_percent: float):
        utilization_percent = max(0.0, min(100.0, utilization_percent))
        self.used_ops = self.max_cap * utilization_percent / 100.0

    def get_utilization(self) -> float:
        if self.max_cap <= 0:
            return 0.0

        return max(0.0, min(1.0, self.used_ops / self.max_cap))

    def get_current_frequency(self) -> float:
        utilization = self.get_utilization()
        return self.min_frequency + utilization * (
            self.max_frequency - self.min_frequency
        )

    def get_current_voltage(self) -> float:
        utilization = self.get_utilization()
        return self.min_voltage + utilization * (
            self.max_voltage - self.min_voltage
        )


# -------------------------------------------------------------------------
# Parseo de la curva SPEC
# -------------------------------------------------------------------------

def parse_spec_power_curve(curve_text: str):
    """
    Convierte un string como:

        0:150,10:265,20:315,30:362,40:413,50:462,
        60:504,70:546,80:589,90:680,100:800

    en:

        [
            (0, 150),
            (10, 265),
            ...
            (100, 800)
        ]
    """

    if curve_text is None:
        return None

    curve = []

    try:
        pairs = curve_text.split(",")

        for pair in pairs:
            utilization, power = pair.split(":")
            utilization = int(utilization.strip())
            power = float(power.strip())
            curve.append((utilization, power))

    except ValueError as exc:
        raise ValueError(
            "Formato incorrecto para --spec-power-curve. "
            "Usa este formato: "
            "0:150,10:265,20:315,30:362,40:413,50:462,"
            "60:504,70:546,80:589,90:680,100:800"
        ) from exc

    expected_utilizations = list(range(0, 101, 10))
    received_utilizations = [u for u, _ in curve]

    if received_utilizations != expected_utilizations:
        raise ValueError(
            "La curva SPEC debe tener exactamente los puntos "
            "0,10,20,30,40,50,60,70,80,90,100 en ese orden."
        )

    return curve


# -------------------------------------------------------------------------
# Creación de modelos
# -------------------------------------------------------------------------

def create_model(model_name: str, alpha: float):
    model_name = model_name.lower()

    if model_name == "linear":
        return EnergyModelServer()

    if model_name == "powerlaw":
        return EnergyModelServerPowerLaw(alpha=alpha)

    if model_name == "empirical":
        return EnergyModelServerEmpirical(alpha=alpha)

    if model_name == "frequency":
        return EnergyModelServerFrequency()

    if model_name == "spec":
        return EnergyModelServerSpecLinearInterpolation()

    raise ValueError(f"Modelo no reconocido: {model_name}")


# -------------------------------------------------------------------------
# Ejecución de pruebas
# -------------------------------------------------------------------------

def run_model(
    model_name: str,
    alpha: float,
    min_power: float,
    max_power: float,
    spec_power_curve=None,
):
    model_name = model_name.lower()

    if model_name == "spec":
        if spec_power_curve is None:
            raise ValueError(
                "Para ejecutar el modelo SPEC debes pasar --spec-power-curve."
            )

        # En SPEC, la curva ya contiene la potencia total real.
        # Por eso sacamos min y max directamente de la curva.
        min_power = spec_power_curve[0][1]
        max_power = spec_power_curve[-1][1]

    server = FakeServer(
        max_power=max_power,
        min_power=min_power,
        max_cap=100.0,
        spec_power_curve=spec_power_curve,
    )

    model = create_model(model_name, alpha)
    model.set_parent(server)

    results = []

    for utilization in UTILIZATION_POINTS:
        server.set_utilization_percent(utilization)

        measurement = model.measure()
        total_power = measurement.total()

        results.append(
            {
                "model": model_name,
                "utilization_percent": utilization,
                "dynamic_power_w": measurement.dynamic,
                "idle_power_w": measurement.idle,
                "total_power_w": total_power,
            }
        )

    return results


# -------------------------------------------------------------------------
# Salida por pantalla y CSV
# -------------------------------------------------------------------------

def print_results_as_csv(results):
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=CSV_COLUMNS,
        lineterminator="\n",
    )

    writer.writeheader()

    for row in results:
        writer.writerow(
            {
                "model": row["model"],
                "utilization_percent": row["utilization_percent"],
                "dynamic_power_w": row["dynamic_power_w"],
                "idle_power_w": row["idle_power_w"],
                "total_power_w": row["total_power_w"],
            }
        )


def save_csv(results, output_path: str):
    output_path = Path(output_path)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "model": row["model"],
                    "utilization_percent": row["utilization_percent"],
                    "dynamic_power_w": row["dynamic_power_w"],
                    "idle_power_w": row["idle_power_w"],
                    "total_power_w": row["total_power_w"],
                }
            )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prueba modelos energéticos de servidor desde 0% hasta 100% "
            "de utilización, devolviendo únicamente puntos de 10 en 10."
        )
    )

    parser.add_argument(
        "--model",
        choices=["linear", "powerlaw", "empirical", "frequency", "spec", "all"],
        help="Modelo a ejecutar.",
    )

    parser.add_argument(
        "--min-power",
        type=float,
        default=100.0,
        help="Potencia mínima o idle power del servidor en W. Default: 100.",
    )

    parser.add_argument(
        "--max-power",
        type=float,
        default=250.0,
        help="Potencia máxima del servidor en W. Default: 250.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Valor alpha para los modelos powerlaw y empirical. Default: 2.0.",
    )

    parser.add_argument(
        "--spec-power-curve",
        default=None,
        help=(
            "Curva SPEC en formato: "
            "0:150,10:265,20:315,30:362,40:413,50:462,"
            "60:504,70:546,80:589,90:680,100:800"
        ),
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Ruta opcional para guardar los resultados en CSV.",
    )

    args = parser.parse_args()

    model = args.model

    if model is None:
        print("Elige un modelo:")
        print("1 - linear")
        print("2 - powerlaw")
        print("3 - empirical")
        print("4 - frequency")
        print("5 - spec")
        print("6 - all")

        option = input("Opción: ").strip()

        options = {
            "1": "linear",
            "2": "powerlaw",
            "3": "empirical",
            "4": "frequency",
            "5": "spec",
            "6": "all",
        }

        model = options.get(option)

        if model is None:
            raise ValueError("Opción no válida.")

    if args.alpha <= 0:
        raise ValueError("--alpha debe ser mayor que 0.")

    if args.min_power < 0:
        raise ValueError("--min-power no puede ser negativo.")

    if args.max_power <= 0:
        raise ValueError("--max-power debe ser mayor que 0.")

    if args.max_power < args.min_power:
        raise ValueError("--max-power debe ser mayor o igual que --min-power.")

    spec_power_curve = parse_spec_power_curve(args.spec_power_curve)

    if model == "all":
        models = ["linear", "powerlaw", "empirical", "frequency", "spec"]
    else:
        models = [model]

    all_results = []

    for model_name in models:
        results = run_model(
            model_name=model_name,
            alpha=args.alpha,
            min_power=args.min_power,
            max_power=args.max_power,
            spec_power_curve=spec_power_curve,
        )

        all_results.extend(results)

    print_results_as_csv(all_results)

    if args.csv:
        save_csv(all_results, args.csv)


if __name__ == "__main__":
    main()