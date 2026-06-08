import numpy as np

from mintedge import Service


"""SIMULATION"""
USE_PREDICTOR = False
MEASUREMENT_INTERVAL = 1  # seconds
ORCHESTRATOR_INTERVAL = 60  # how often the orchestrator updates the allocation
CAPACITY_BUFFER = 0.2  # [0, 1] share of extra capacity to allocate
REACTIVE_ALLOCATION = False  # whether to allocate resources reactively when more than a threshold of requests are rejected
REACTION_THRESHOLD = 0.1  # share of reqs rejected to trigger a new allocation

"""SCENARIO"""
PLOT_SCENARIO = False

API_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Some examples of coordinates
# Twente's coordinates -->  funciona
# NORTH, SOUTH, EAST, WEST = 52.4914, 52.1175, 7.0827, 6.3264

# Enschede + Hengelo's coordinates --> funciona
# NORTH, SOUTH, EAST, WEST = 52.2978, 52.1796, 6.9519, 6.7456

# Elburg's coordinates --> funciona
NORTH, SOUTH, EAST, WEST = 52.4788, 52.3525, 5.9268, 5.7536

# Maastrichts's coordinates--> funciona
# NORTH, SOUTH, EAST, WEST = 50.8695, 50.8303, 5.7417, 5.6415

# Luxembourg's state coordinates--> funciona
# NORTH, SOUTH, EAST, WEST = 50.1848, 49.4457, 6.5341, 5.7307

# Luxembourg's city coordinates --> crea el mapa, pero da error
# NORTH, SOUTH, EAST, WEST = 49.7575, 49.4139, 6.45978, 5.75931

PROVIDER = "vodafone"
BSS_FILE = "./scenario/bss.csv"
LINKS_FILE = "./scenario/links.csv"

RANDOM_ROUTES = True
# If ROUTES_FILE is provided, the NET_FILE used to generate it must be provided too
NET_FILE = "./scenario/Luxembourg.net.xml"
# If RANDOM_ROUTES is True, ROUTES_FILE is ignored
ROUTES_FILE = "./scenario/Luxembourg.rou.xml"

# The number of cars, pedestrians and stationary users is only considered if
# RANDOM_ROUTES is True
NUMBER_OF_CARS = 500
NUMBER_OF_PEOPLE = 100
NUMBER_OF_STATIONARY = 150


# The user count distribution expresses the share of active users over the total
# for each hour of the day. You can combine this with RANDOM_ROUTES to generate
# dynamic user counts.
# fmt: off
USER_COUNT_DISTRIBUTION = [0.04, 0.2, 0.5, 0.33, 0.2, 0.02, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.4, 0.2, 0.1, 0.02]
# fmt: on

"""BASE STATION"""
BS_BANDWIDTH = 100e6  # 100 MHz (METIS-II Table 3-9 UC5 (connected cars))
SNR0_DB = 55  # reference SNR at 1 meter (dB)
SNR0_LIN = 10 ** (SNR0_DB / 10)  # linear SNR at 1 meter
PATHLOSS_EXPONENT = 2.0  # free space
MIN_USER_RATE = 1e6  # 1 Mbps
# Shannon-Hartley theorem. This is the base datarate used for preallocations
BS_DATARATE = BS_BANDWIDTH * np.log2(1.0 + SNR0_LIN)


"""BACKHAUL"""
W_PER_BIT = 5.9e-9  # 5.9 nJ/bit
MAX_LINK_CAPACITY = 10e9  # 10 Gbps

"""EDGE SERVERS"""
SHARE_OF_SERVERS = 0.5

SERVERS = [
    # {  # HP ProLiant DL560 Gen11 Intel Xeon Platinum 8490H 1.90 GHz
    #     "MAX_POWER": 1280,
    #     "IDLE_POWER": 415,
    #     "MAX_CAPACITY": 22151384,
    #     "BOOT_TIME": 20,

    #     "CAPACITANCE": ,
    #     "ACTIVITY_FACTOR": ,
    #     "FREQUENCIES": ,
    #     "VOLTAGES": ,
    # },

    #{  # Transmeta Crusoe TM5800 
    #    "MAX_POWER": 15,
    #    "IDLE_POWER": 1.0,
    #    "MAX_CAPACITY": 1000000000,
    #    "BOOT_TIME": 1,

    #    "TOTAL_CORES": 1,
    #    "MIN_FREQUENCY": 300e6,
    #    "MAX_FREQUENCY": 1000e6,

    #   "CAPACITANCE": 4.3e-9,  
    #   "ACTIVITY_FACTOR": 0.1,  
    #    "FREQUENCIES": [300e6, 433e6, 533e6, 667e6, 800e6, 900e6, 1000e6],
    #    "VOLTAGES": [0.80, 0.875, 0.95, 1.05, 1.15, 1.25, 1.30],
    # },

    #{  # HP ProLiant DL380a Gen11 Intel Xeon Platinum 8480+ base=2.0 GHz, turbo=3.8 GHz, nominal=2.5 GHz
    #    "MAX_POWER": 696,
    #    "IDLE_POWER": 222,
    #    "MAX_CAPACITY": 11260532,
    #   "BOOT_TIME": 20,

    #    "CAPACITANCE": 318e-9,  # creado a partir de la fórmula P = P_idle + A * C * V^2 * f, con p_idle=222,P_max=696, A=0.2, V=1.4 y f=3,8e9
    #    "ACTIVITY_FACTOR": 0.2,  # creado a partir de la fórmula P = P_idle + A * C * V^2 * f, con P_idle=222, P_max=696, C=318e-9, V=1.4 y f=3.8e9
    #    "FREQUENCIES": [2.0e9, 2.4e9, 2.8e9, 3.2e9, 3.8e9],  # inventado, menos el primer valor que es el base clock y el último que es el turbo clock
    #    "VOLTAGES": [1.0, 1.1, 1.2, 1.3, 1.4],  # inventado
    #},

    #{  # Intel Pentium M 1.4 GHz - DVS / SpeedStep, basado en Ge et al. 2005
    #    "MAX_POWER": 59,          # sistema completo aproximado en el ejemplo medido
    #    "IDLE_POWER": 38,         # estimado, no aparece directamente en el paper
    #    "MAX_CAPACITY": 1000000,  # valor a adaptar al simulador
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": 33e-9,     # estimada, no medida directamente en el paper -->A*C
    #    "ACTIVITY_FACTOR": 0.2,   # supuesto razonable --> A*C
    #    "FREQUENCIES": [600e6, 800e6, 1.0e9, 1.2e9, 1.4e9],
    #    "VOLTAGES": [0.956, 1.180, 1.308, 1.436, 1.484],
    #},

    {  # TI OMAP3530 / OMAP35x - ARM Cortex-A8 arquitectura 0.65 mm
        "MAX_POWER": 1.679,       
        "IDLE_POWER": 0.0069,      
        "MAX_CAPACITY": 10000000,      
        "BOOT_TIME": 20,

        "CAPACITANCE": 0.52e-9,   # Ceff = A*C, derivado de 0.52 mA/MHz/V
        "ACTIVITY_FACTOR": 1.0,   # Ceff ya incluye actividad efectiva
        "FREQUENCIES": [125e6, 250e6, 500e6, 550e6, 600e6],
        "VOLTAGES": [0.95, 1.00, 1.20, 1.27, 1.35],
    },


    # {  # AMD Opteron 6174 - 12 cores, 2.2 GHz
    #     "MAX_POWER": 115,
    #     "IDLE_POWER": 53,
    #     "MAX_CAPACITY": 2654268,
    #     "BOOT_TIME": 20,

    #     "CAPACITANCE": 1.996163e-7,  # calibrado; con A=0.1 da A*C=1.996163e-8
    #     "ACTIVITY_FACTOR": 0.1,  # asumido
    #     "FREQUENCIES": [2.2e9, 1.8e9, 1.4e9, 1.1e9, 0.8e9],
    #     "VOLTAGES": [1.1875, 1.1375, 1.1125, 1.0875, 1.0625],
    # },

    # {  # AMD Opteron 8356 - 4 cores, 2.3 GHz
    #     "MAX_POWER": 115,
    #     "IDLE_POWER": 38,
    #     "MAX_CAPACITY": 924972,
    #     "BOOT_TIME": 20,

    #     "CAPACITANCE": 2.132531e-7,  # calibrado; con A=0.1 da A*C=2.132531e-8
    #     "ACTIVITY_FACTOR": 0.1,  # asumido
    #     "FREQUENCIES": [2.3e9, 2.0e9, 1.7e9, 1.4e9, 1.15e9],
    #     "VOLTAGES": [1.250, 1.150, 1.100, 1.050, 1.050],
    # },

    # {  # FUJITSU Server PRIMERGY CX2560 M7 PRIMERGY CX400 M6
    #     "MAX_POWER": 2336,
    #     "IDLE_POWER": 541,
    #     "MAX_CAPACITY": 33244766,
    #     "BOOT_TIME": 20,

    #     "MAX_FREQUENCY": ,
    #     "MIN_FREQUENCY": ,
    #     "TOTAL_CORES": ,
    # },
]
# Data from OpenSpecPower

"""SERVICES"""
CAR_SERVICES = ["connected_vehicles"]
PEDESTRIAN_SERVICES = ["augmented_reality", "virtual_reality"]
STATIONARY_SERVICES = ["video_analysis"]

SERVICES = [
    # name, workload(ops/s), lambda(req/s), vin(bytes), vout(bytes), delay_budget(seconds)
    #
    # CONNECTED VEHICLES
    Service("connected_vehicles", 14000, 10, 1600, 100, 5e-3),
    # AUGMENTED REALITY
    Service("augmented_reality", 50000, 0.5, 1500 * 1024, 25 * 1024, 15e-3),
    # VIDEO ANALYSIS
    Service("video_analysis", 30000, 6, 1500 * 1024, 20, 30e-3),
]

# -------------------ENERGY MODELS---------------------------
#------------------------------------------------------------
# linear model: P = P_idle + (P_max - P_idle) * (used_capacity / max_capacity)
# power law model: P = P_idle + (P_max - P_idle) * (used_capacity / max_capacity) ** alpha
# polynomial model: P = P_idle + (P_max - P_idle) * (a * (used_capacity / max_capacity) ** 2 + b * (used_capacity / max_capacity) + c)
# frequency-based model: P = P_idle + A * C * V^2 * f, where A is the activity factor, C is the capacitance, V is the voltage and f is the frequency. In this case, we can model the power consumption as a function of the frequency, which is a common approach in DVFS (Dynamic Voltage and Frequency Scaling) techniques.


# modelos posibles: "linear", "powerlaw"(pasarle alpha), "polynomial"(pasarle alpha), "frequency"

SERVER_ENERGY_MODEL = "linear"
ALPHA = 1