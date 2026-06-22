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
NUMBER_OF_CARS = 2500
NUMBER_OF_PEOPLE = 1000
NUMBER_OF_STATIONARY = 500


# The user count distribution expresses the share of active users over the total
# for each hour of the day. You can combine this with RANDOM_ROUTES to generate
# dynamic user counts.
# fmt: off
USER_COUNT_DISTRIBUTION = [0.3, 0.4, 0.6, 0.5, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.4, 0.2, 0.1, 0.02]
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
SHARE_OF_SERVERS = 0.04

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


    #{  # HP ProLiant DL380a Gen11 Intel Xeon Platinum 8480+ base=2.0 GHz, turbo=3.8 GHz, nominal=2.5 GHz
    #    "MAX_POWER": 696,
    #    "IDLE_POWER": 222,
    #    "MAX_CAPACITY": 11260532,
    #    "BOOT_TIME": 20,

    #    "SPEC_POWER_CURVE": [
    #        (0, 222),
    #        (10, 293),
    #        (20, 327),
    #        (30, 361),
    #        (40, 393),
    #        (50, 426),
    #        (60, 462),
    #        (70, 503),
    #        (80, 551),
    #        (90, 619),
    #        (100, 696),
    #    ],
    #    "CAPACITANCE": 318e-9,  # creado a partir de la fórmula P = P_idle + A * C * V^2 * f, con p_idle=222,P_max=696, A=0.2, V=1.4 y f=3,8e9
    #    "ACTIVITY_FACTOR": 0.2,  # creado a partir de la fórmula P = P_idle + A * C * V^2 * f, con P_idle=222, P_max=696, C=318e-9, V=1.4 y f=3.8e9
    #    "FREQUENCIES": [2.0e9, 2.4e9, 2.8e9, 3.2e9, 3.8e9],  # inventado, menos el primer valor que es el base clock y el último que es el turbo clock
    #    "VOLTAGES": [1.0, 1.1, 1.2, 1.3, 1.4],  # inventado
    #},

    { # ASUSTeK RS720A-E13-RS8U - AMD EPYC 9965, 2 chips, 384 cores total
        "MAX_POWER": 800,
        "IDLE_POWER": 150,
        "MAX_CAPACITY": 39140079,  # SPECpower_ssj2008 ssj_ops @ 100%
        "BOOT_TIME": 20,
        "SPEC_POWER_CURVE": [
            (0, 150),
            (10, 265),
            (20, 315),
            (30, 362),
            (40, 413),
            (50, 462),
            (60, 504),
            (70, 546),
            (80, 589),
            (90, 680),
            (100, 800),
        ],
        "ACTIVITY_FACTOR": 1.0,
        "CAPACITANCE": None,
        "FREQUENCIES": [1.5e9, 2.0e9, 2.5e9, 3.0e9],  # inventado´
        "VOLTAGES": [1.0, 1.1, 1.2, 1.3, 1.4],  # inventado
    },

    #{  # Supermicro SuperWorkstation 5039A-i Intel Xeon W-2123 
    #    "MAX_POWER": 28.21,
    #    "IDLE_POWER": 0,
    #    "MAX_CAPACITY": 274000,  # normalizado desde SPEC CPU2017: SPECrate2017_int_base = 27.4 -> 27.4 * 10000
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": 8.2e-9,  # Ceff del artículo: Ceff = 8.2e-9 F.
    #    "ACTIVITY_FACTOR": 1.0,  # se deja a 1 porque la capacitancia ya es Ceffç
    #    "FREQUENCIES": [
    #        1.2e9, 1.4e9, 1.6e9, 1.8e9, 2.0e9, 2.2e9, 2.4e9,
    #        2.6e9, 2.8e9, 3.0e9, 3.2e9, 3.4e9, 3.6e9
    #    ],  # pasos de 200 MHz dentro del rango medido 1.2-3.6 GHz
    #     "VOLTAGES": [
    #        0.756, 0.774, 0.792, 0.810, 0.828, 0.846, 0.865,
    #        0.883, 0.901, 0.919, 0.937, 0.955, 0.973
    #    ],  # interpolado linealmente entre 0.756 V a 1.2 GHz y 0.973 V a 3.6 GHz
    #},

    #{  # HP ProLiant DL180 G5 Intel Xeon L5420
    #    "MAX_POWER": 189,
    #    "IDLE_POWER": 106,
    #    "MAX_CAPACITY": 282281,  # SPECpower_ssj2008: ssj_ops@100% = 282,281
    #    "BOOT_TIME": 20,    

    #    "CAPACITANCE": 12.6e-9,  # calculado a partir de la suma de la tabla de capacitancias
    #    "ACTIVITY_FACTOR": 1,  # supuesto de actividad completa, ya que la capacitancia ya es Ceff
    #    "FREQUENCIES": [2.0e9, 2.5e9],  # del artículo: dos P-states, 2.0 GHz y 2.5 GHz
    #    "VOLTAGES": [1.104, 1.104],  # del artículo: Intel SpeedStep a 2.0 GHz y 2.5 GHz usa 1.104 V
    #},
    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------
    #{  # Supermicro 2021M-UR+ AMD Opteron 2380, 2 chips, 8 cores total
    #    "MAX_POWER": 269,
    #    "IDLE_POWER": 138,
    #    "MAX_CAPACITY": 308089,  # SPECpower_ssj2008
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": None,  # no encontrado
    #    "ACTIVITY_FACTOR": 1,
    #    "FREQUENCIES": [2.5e9, 1.8e9, 1.3e9, 0.8e9],
        # AMD da rangos, uso el punto medio:
        # P0: 1.150-1.325 V, P1: 1.050-1.225 V,
        # P2: 0.950-1.125 V, P3: 0.875-1.050 V
        #"VOLTAGES": [1.2375, 1.1375, 1.0375, 0.9625],
    #},

    #{  # HP ProLiant DL385 G5p AMD Opteron 2384, 2 chips, 8 cores total
    #    "MAX_POWER": 257,
    #    "IDLE_POWER": 147,
    #    "MAX_CAPACITY": 341306,  # SPECpower_ssj2008: ssj_ops@100% = 341,306
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": None,  # no encontrado
    #    "ACTIVITY_FACTOR": 1,
    #    "FREQUENCIES": [2.7e9, 2.0e9, 1.5e9, 0.8e9],
        # AMD da rangos, uso el punto medio:
        # P0: 1.150-1.325 V, P1: 1.050-1.225 V,
        # P2: 0.950-1.125 V, P3: 0.850-1.025 V
        #"VOLTAGES": [1.2375, 1.1375, 1.0375, 0.9375],
    #   },

    #{  # HP ProLiant DL385 G6 AMD Opteron 2435, 2 chips, 12 cores total
    #    "MAX_POWER": 260,
    #    "IDLE_POWER": 124,
    #    "MAX_CAPACITY": 535814,  # SPECpower_ssj2008: ssj_ops@100% = 535,814
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": 24e-9,  # no encontrado x1-->18.55e-9
    #    "ACTIVITY_FACTOR": 1,
    #    "FREQUENCIES": [0.8e9, 1.4e9, 1.7e9, 2.1e9, 2.6e9],
        # AMD da rangos
        # P0: 1.075-1.300 V, P1: 1.025-1.250 V,
        # P2: 1.000-1.225 V, P3: 0.975-1.200 V,
        # P4: 0.900-1.125 V
    #    "VOLTAGES": [1.125, 1.2, 1.225, 1.250, 1.3],
    #},
    #----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------

    #{  # TI OMAP3530 / OMAP35x - ARM Cortex-A8 arquitectura 0.65 mm
    #    "MAX_POWER": 1.679,       
    #    "IDLE_POWER": 0.0069,      
    #    "MAX_CAPACITY": 10000000,      
    #    "BOOT_TIME": 20,

    #    "CAPACITANCE": 0.52e-9,   # Ceff = A*C, derivado de 0.52 mA/MHz/V
    #    "ACTIVITY_FACTOR": 1.0,   # Ceff ya incluye actividad efectiva
    #    "FREQUENCIES": [125e6, 250e6, 500e6, 550e6, 600e6],
    #    "VOLTAGES": [0.95, 1.00, 1.20, 1.27, 1.35],
    #},


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
    Service("connected_vehicles", 1000, 10, 1600, 100, 5e-3), #14000
    # AUGMENTED REALITY
    Service("augmented_reality", 3000, 0.5, 1500 * 1024, 25 * 1024, 15e-3), #50000
    # VIDEO ANALYSIS
    Service("video_analysis", 2000, 6, 1500 * 1024, 20, 30e-3), #30000
]

# -------------------ENERGY MODELS---------------------------
#------------------------------------------------------------
# linear model: P = P_idle + (P_max - P_idle) * utilization
# power law model: P = P_idle + (P_max - P_idle) * utilization ** alpha
# empirical model: P = P_idle + (P_max - P_idle) * (2 * utilization - utilization ** alpha) --> (alpha == r ) --> r suele ser 1.4
# frequency model: P = P_idle + A * C * V^2 * f -- > DSFV
# spec_linear_interpolation: P = f(utilization) =  --> f is a piecewise linear function defined by the SPECpower_ssj2008 measurements at different utilization levels

# Alpha i s a parameter of calibration for the power law and empirical models.                                                     
# modelos posibles: "linear", "powerlaw"(pasarle alpha), "empirical"(pasarle alpha), "frequency" , "spec_linear_interpolation" 
SERVER_ENERGY_MODEL = "spec_linear_interpolation"
ALPHA = 1.4
