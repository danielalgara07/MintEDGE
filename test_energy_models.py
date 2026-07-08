import argparse
import csv
import importlib.util
import sys
from pathlib import Path


# -------------------------------------------------------------------------
# Columnas del CSV
# -------------------------------------------------------------------------

CSV_COLUMNS = [
    "model",
    "alpha",
    "utilization_percent",
    "frequency_hz",
    "voltage_v",
    "ceff",
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
# Funciones auxiliares
# -------------------------------------------------------------------------

def parse_float(value: str) -> float:
    """
    Permite escribir decimales con punto o con coma.

    Ejemplos:
        2.5
        2,5
        1.79e-7
        1,79e-7
    """

    return float(str(value).replace(",", "."))


def parse_float_list(text: str):
    """
    Convierte un texto en una lista de floats.

    Se usa ; como separador para no crear conflicto con la coma decimal.

    Ejemplos:
        "0;1.2;1.8;2.4;3.0"
        "0;1,2;1,8;2,4;3,0"
        "0.75;0.85;0.95;1.05;1.10"
        "0,75;0,85;0,95;1,05;1,10"
    """

    if text is None:
        return None

    values = []

    for item in str(text).split(";"):
        item = item.strip()

        if item:
            values.append(parse_float(item))

    return values


def convertir_frecuencias_a_hz(frequencies, unit):
    """
    Convierte una lista de frecuencias a Hz.
    """

    if frequencies is None:
        return None

    unit = str(unit).strip().lower()

    if unit == "hz":
        factor = 1.0
    elif unit == "mhz":
        factor = 1.0e6
    elif unit == "ghz":
        factor = 1.0e9
    else:
        raise ValueError(
            "Unidad de frecuencia no válida. Usa hz, mhz o ghz."
        )

    return [frequency * factor for frequency in frequencies]


def format_number(value):
    """
    Evita que la utilización salga como 10.0 cuando realmente es 10.
    Si es decimal, se mantiene como decimal.
    """

    value = round(float(value), 10)

    if value.is_integer():
        return int(value)

    return value


def format_optional_number(value):
    """
    Formatea números opcionales para escribirlos limpios en el CSV.
    """

    if value is None or value == "":
        return ""

    value = float(value)

    if value.is_integer():
        return int(value)

    return f"{value:g}"


def generate_utilization_points(step: float):
    """
    Genera los puntos de utilización desde 0 hasta 100.

    Ejemplo:
        step = 10   -> 0, 10, 20, ..., 100
        step = 2.5  -> 0, 2.5, 5, 7.5, ..., 100
    """

    points = []
    current = 0.0

    while current <= 100.0:
        points.append(format_number(current))
        current += step

    if points[-1] != 100:
        points.append(100)

    return points


def interpolate_value_by_utilization(values, utilization_percent: float):
    """
    Interpola un valor de una lista según la utilización.

    Si hay 5 valores, se interpretan así:
        0%, 25%, 50%, 75%, 100%

    Si hay 11 valores, se interpretan así:
        0%, 10%, 20%, ..., 100%
    """

    if values is None or len(values) == 0:
        raise ValueError("La lista de valores está vacía.")

    if len(values) == 1:
        return float(values[0])

    utilization_percent = max(0.0, min(100.0, float(utilization_percent)))

    position = utilization_percent / 100.0 * (len(values) - 1)

    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)

    fraction = position - lower_index

    lower_value = float(values[lower_index])
    upper_value = float(values[upper_index])

    return lower_value + fraction * (upper_value - lower_value)


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

    Para el modelo frequency se puede trabajar de dos formas:

    1. Sin pasar --ceff, --frequencies ni --voltages:
       se mantiene el comportamiento anterior y se calibra automáticamente
       activity_factor para que al 100% se llegue aproximadamente a max_power.

    2. Pasando --ceff, --frequencies y --voltages:
       se usa la fórmula corregida:
           P_dynamic = Ceff * (V_actual^2 * f_actual - V_idle^2 * f_idle)

       Así en el 0%:
           P_dynamic = 0
           P_total = P_idle
    """

    def __init__(
        self,
        max_power=250.0,
        min_power=100.0,
        max_cap=100.0,
        spec_power_curve=None,
        frequency_values=None,
        voltage_values=None,
        ceff=None,
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

        # Valores personalizados para el modelo frequency.
        self.frequency_values = frequency_values
        self.voltage_values = voltage_values

        if self.frequency_values is not None or self.voltage_values is not None:
            if self.frequency_values is None or self.voltage_values is None:
                raise ValueError(
                    "Debes pasar tanto --frequencies como --voltages."
                )

            if len(self.frequency_values) != len(self.voltage_values):
                raise ValueError(
                    "--frequencies y --voltages deben tener la misma longitud."
                )

        # Valores por defecto usados si no se pasa una lista personalizada.
        self.min_frequency = 0.0
        self.max_frequency = 3.0e9

        self.min_voltage = 0.75
        self.max_voltage = 1.10

        if self.frequency_values is not None:
            self.min_frequency = min(self.frequency_values)
            self.max_frequency = max(self.frequency_values)

        if self.voltage_values is not None:
            self.min_voltage = min(self.voltage_values)
            self.max_voltage = max(self.voltage_values)

        # Parámetros usados por EnergyModelServerFrequency.
        if ceff is not None:
            # Con ceff personalizado:
            # Como ceff ya agrupa actividad y capacitancia:
            # activity_factor = 1
            self.capacitance = float(ceff)
            self.activity_factor = 1.0
            self.uses_custom_ceff = True
        else:
            # Comportamiento anterior: se calibra automáticamente para que
            # al 100% la potencia total sea aproximadamente max_power.
            self.capacitance = 45e-9
            self.uses_custom_ceff = False

            dynamic_target = max(self.max_power - self.idle_power, 0.0)
            denominator = (
                self.capacitance
                * (self.max_voltage ** 2)
                * self.max_frequency
            )

            if denominator > 0:
                self.activity_factor = dynamic_target / denominator
            else:
                self.activity_factor = 0.0

    def set_utilization_percent(self, utilization_percent: float):
        utilization_percent = max(0.0, min(100.0, float(utilization_percent)))
        self.used_ops = self.max_cap * utilization_percent / 100.0

    def get_utilization(self) -> float:
        if self.max_cap <= 0:
            return 0.0

        return max(0.0, min(1.0, self.used_ops / self.max_cap))

    def get_current_frequency(self) -> float:
        utilization_percent = self.get_utilization() * 100.0

        if self.frequency_values is not None:
            return interpolate_value_by_utilization(
                self.frequency_values,
                utilization_percent,
            )

        utilization = self.get_utilization()

        return self.min_frequency + utilization * (
            self.max_frequency - self.min_frequency
        )

    def get_current_voltage(self) -> float:
        utilization_percent = self.get_utilization() * 100.0

        if self.voltage_values is not None:
            return interpolate_value_by_utilization(
                self.voltage_values,
                utilization_percent,
            )

        utilization = self.get_utilization()

        return self.min_voltage + utilization * (
            self.max_voltage - self.min_voltage
        )

    def get_reference_frequency(self) -> float:
        """
        Frecuencia de referencia para el estado idle.

        Si se han pasado frecuencias personalizadas, se usa la primera.
        Por eso las frecuencias deben pasarse de menor a mayor carga.
        """

        if self.frequency_values is not None:
            return float(self.frequency_values[0])

        return float(self.min_frequency)

    def get_reference_voltage(self) -> float:
        """
        Voltaje de referencia para el estado idle.

        Si se han pasado voltajes personalizados, se usa el primero.
        Por eso los voltajes deben pasarse de menor a mayor carga.
        """

        if self.voltage_values is not None:
            return float(self.voltage_values[0])

        return float(self.min_voltage)

    def get_relative_frequency_dynamic_power(self) -> float:
        """
        Calcula la potencia dinámica relativa al estado idle.

        En vez de:
            P_dynamic = Ceff * V^2 * f

        usa:
            P_dynamic = Ceff * (V^2 * f - V_idle^2 * f_idle)

        Así se evita sumar dos veces la parte de consumo que ya está incluida
        dentro de idle_power.
        """

        current_frequency = self.get_current_frequency()
        current_voltage = self.get_current_voltage()

        reference_frequency = self.get_reference_frequency()
        reference_voltage = self.get_reference_voltage()

        current_term = (current_voltage ** 2) * current_frequency
        reference_term = (reference_voltage ** 2) * reference_frequency

        dynamic_power = self.capacitance * self.activity_factor * (
            current_term - reference_term
        )

        return max(0.0, dynamic_power)


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
            power = parse_float(power.strip())
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
    step: float,
    spec_power_curve=None,
    frequency_values=None,
    voltage_values=None,
    ceff=None,
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
        frequency_values=frequency_values,
        voltage_values=voltage_values,
        ceff=ceff,
    )

    model = create_model(model_name, alpha)
    model.set_parent(server)

    results = []
    utilization_points = generate_utilization_points(step)

    for utilization in utilization_points:
        server.set_utilization_percent(utilization)

        # -------------------------------------------------------------
        # Corrección del modelo frequency con ceff personalizado
        # -------------------------------------------------------------
        #
        # Si el usuario ha pasado --ceff, se usa:
        #   dynamic = Ceff * (V_actual^2*f_actual - V_idle^2*f_idle)
        #
        # Así:
        #   en 0% -> dynamic = 0
        #   total = idle_power
        #
        # Para el resto de modelos, se usa model.measure() normalmente.
        # -------------------------------------------------------------

        measurement = model.measure()
        dynamic_power = measurement.dynamic
        idle_power = measurement.idle
        total_power = measurement.total()

        if model_name in {"powerlaw", "empirical"}:
            alpha_value = alpha
        else:
            alpha_value = ""

        if model_name == "frequency":
            frequency_hz = server.get_current_frequency()
            voltage_v = server.get_current_voltage()
            ceff_value = server.capacitance
        else:
            frequency_hz = ""
            voltage_v = ""
            ceff_value = ""

        results.append(
            {
                "model": model_name,
                "alpha": alpha_value,
                "utilization_percent": utilization,
                "frequency_hz": frequency_hz,
                "voltage_v": voltage_v,
                "ceff": ceff_value,
                "dynamic_power_w": dynamic_power,
                "idle_power_w": idle_power,
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
                "alpha": format_optional_number(row["alpha"]),
                "utilization_percent": row["utilization_percent"],
                "frequency_hz": format_optional_number(row["frequency_hz"]),
                "voltage_v": format_optional_number(row["voltage_v"]),
                "ceff": format_optional_number(row["ceff"]),
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
                    "alpha": format_optional_number(row["alpha"]),
                    "utilization_percent": row["utilization_percent"],
                    "frequency_hz": format_optional_number(row["frequency_hz"]),
                    "voltage_v": format_optional_number(row["voltage_v"]),
                    "ceff": format_optional_number(row["ceff"]),
                    "dynamic_power_w": row["dynamic_power_w"],
                    "idle_power_w": row["idle_power_w"],
                    "total_power_w": row["total_power_w"],
                }
            )


# -------------------------------------------------------------------------
# Validación de parámetros de frequency
# -------------------------------------------------------------------------

def preparar_parametros_frequency(args):
    """
    Lee y valida los parámetros personalizados del modelo frequency.

    Si no se pasa ninguno, devuelve valores None y se usa el comportamiento
    por defecto del modelo frequency.
    """

    custom_frequency_used = (
        args.ceff is not None
        or args.frequencies is not None
        or args.voltages is not None
    )

    if not custom_frequency_used:
        return None, None, None

    if args.ceff is None:
        raise ValueError(
            "Si usas --frequencies o --voltages, también debes pasar --ceff."
        )

    if args.frequencies is None:
        raise ValueError(
            "Si usas --ceff o --voltages, también debes pasar --frequencies."
        )

    if args.voltages is None:
        raise ValueError(
            "Si usas --ceff o --frequencies, también debes pasar --voltages."
        )

    ceff = parse_float(args.ceff)

    if ceff <= 0:
        raise ValueError("--ceff debe ser mayor que 0.")

    raw_frequencies = parse_float_list(args.frequencies)
    voltages = parse_float_list(args.voltages)

    if raw_frequencies is None or len(raw_frequencies) == 0:
        raise ValueError("--frequencies no puede estar vacío.")

    if voltages is None or len(voltages) == 0:
        raise ValueError("--voltages no puede estar vacío.")

    frequencies = convertir_frecuencias_a_hz(
        frequencies=raw_frequencies,
        unit=args.frequency_unit,
    )

    if len(frequencies) != len(voltages):
        raise ValueError(
            "--frequencies y --voltages deben tener la misma longitud."
        )

    if len(frequencies) < 2:
        raise ValueError(
            "Debes pasar al menos dos frecuencias y dos voltajes para formar un rango."
        )

    for frequency in frequencies:
        if frequency < 0:
            raise ValueError("Las frecuencias no pueden ser negativas.")

    for voltage in voltages:
        if voltage <= 0:
            raise ValueError("Los voltajes deben ser mayores que 0.")

    return frequencies, voltages, ceff


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prueba modelos energéticos de servidor desde 0% hasta 100% "
            "de utilización. El salto se controla con --step."
        )
    )

    parser.add_argument(
        "--model",
        choices=["linear", "powerlaw", "empirical", "frequency", "spec", "all"],
        help="Modelo a ejecutar.",
    )

    parser.add_argument(
        "--min-power",
        type=parse_float,
        default=100.0,
        help="Potencia mínima o idle power del servidor en W. Default: 100.",
    )

    parser.add_argument(
        "--max-power",
        type=parse_float,
        default=250.0,
        help="Potencia máxima del servidor en W. Default: 250.",
    )

    parser.add_argument(
        "--alpha",
        type=parse_float,
        default=2.0,
        help="Valor alpha para los modelos powerlaw y empirical. Default: 2.0.",
    )

    parser.add_argument(
        "--step",
        type=parse_float,
        default=10.0,
        help=(
            "Salto de utilización en porcentaje. "
            "Ejemplo: --step 10, --step 5, --step 2.5 o --step 2,5. "
            "Default: 10."
        ),
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
        "--ceff",
        default=None,
        help=(
            "Capacitancia efectiva para el modelo frequency. "
            "Ejemplo: --ceff 2.33e-7"
        ),
    )

    parser.add_argument(
        "--frequencies",
        default=None,
        help=(
            "Lista de frecuencias para el modelo frequency separadas por ;. "
            "Deben ir de menor a mayor carga. "
            "Ejemplo: --frequencies \"0.80;1.00;1.30;1.50;1.90\""
        ),
    )

    parser.add_argument(
        "--frequency-unit",
        choices=["hz", "mhz", "ghz"],
        default="ghz",
        help=(
            "Unidad de las frecuencias pasadas con --frequencies. "
            "Opciones: hz, mhz, ghz. Default: ghz."
        ),
    )

    parser.add_argument(
        "--voltages",
        default=None,
        help=(
            "Lista de voltajes para el modelo frequency separados por ;. "
            "Debe tener la misma longitud que --frequencies. "
            "Deben ir de menor a mayor carga. "
            "Ejemplo: --voltages \"0.97;0.98;1.00;1.03;1.07\""
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

    if args.step <= 0:
        raise ValueError("--step debe ser mayor que 0.")

    if args.step > 100:
        raise ValueError("--step debe ser menor o igual que 100.")

    if args.min_power < 0:
        raise ValueError("--min-power no puede ser negativo.")

    if args.max_power <= 0:
        raise ValueError("--max-power debe ser mayor que 0.")

    if args.max_power < args.min_power:
        raise ValueError("--max-power debe ser mayor o igual que --min-power.")

    spec_power_curve = parse_spec_power_curve(args.spec_power_curve)

    frequency_values, voltage_values, ceff = preparar_parametros_frequency(args)

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
            step=args.step,
            spec_power_curve=spec_power_curve,
            frequency_values=frequency_values,
            voltage_values=voltage_values,
            ceff=ceff,
        )

        all_results.extend(results)

    print_results_as_csv(all_results)

    if args.csv:
        save_csv(all_results, args.csv)


if __name__ == "__main__":
    main()