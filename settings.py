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
# Twente's coordinates -->  funciona vodafone = 195 estaciones, t-mobile = 182, kpn = 230
# NORTH, SOUTH, EAST, WEST = 52.4914, 52.1175, 7.0827, 6.3264

# Enschede + Hengelo's coordinates --> funciona vodafone = 68 estaciones , t-mobile = 74, kpn = 76
# NORTH, SOUTH, EAST, WEST = 52.2978, 52.1796, 6.9519, 6.7456

# Elburg's coordinates --> funciona vodafone = 25 estaciones, t-mobile = 24 estaciones, kpn = 36 estaciones,
NORTH, SOUTH, EAST, WEST = 52.4788, 52.3525, 5.9268, 5.7536

# Maastrichts's coordinates--> funciona vodafone = 41 estaciones
# NORTH, SOUTH, EAST, WEST = 50.8695, 50.8303, 5.7417, 5.6415

# Luxembourg's state coordinates--> funciona pero pone que no encuentra bases en ese area
# NORTH, SOUTH, EAST, WEST = 50.1848, 49.4457, 6.5341, 5.7307

# Luxembourg's city coordinates --> funciona pero pone que no encuentra bases en ese area
# NORTH, SOUTH, EAST, WEST = 49.7575, 49.4139, 6.45978, 5.75931

PROVIDER = "kpn"
BSS_FILE = "./scenario/bss.csv"
LINKS_FILE = "./scenario/links.csv"

RANDOM_ROUTES = True
# If ROUTES_FILE is provided, the NET_FILE used to generate it must be provided too
NET_FILE = "./scenario/Luxembourg.net.xml"
# If RANDOM_ROUTES is True, ROUTES_FILE is ignored
ROUTES_FILE = "./scenario/Luxembourg.rou.xml"

# The number of cars, pedestrians and stationary users is only considered if
# RANDOM_ROUTES is True
NUMBER_OF_CARS = 1000
NUMBER_OF_PEOPLE = 1500
NUMBER_OF_STATIONARY = 10


# The user count distribution expresses the share of active users over the total
# for each hour of the day. You can combine this with RANDOM_ROUTES to generate
# dynamic user counts.
# fmt: off
USER_COUNT_DISTRIBUTION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.6, 0.7, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.4, 0.2, 0.1, 0.02]
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
SHARE_OF_SERVERS = 1.00

SERVERS = [
 
    {  # SERVER 1 - HP ProLiant DL180 G5 Intel Xeon E5450 3.00 GHz
        "MAX_POWER": 244,
        "IDLE_POWER": 123,
        "MAX_CAPACITY": 299610,
        "BOOT_TIME": 20,

        "SPEC_POWER_CURVE": [
            (0, 123),
            (10, 142),
            (20, 158),
            (30, 173),
            (40, 188),
            (50, 201),
            (60, 212),
            (70, 222),
            (80, 230),
            (90, 238),
            (100, 244),
        ],

        "CAPACITANCE": 0.1e-7,
        "ACTIVITY_FACTOR": 1,

        "FREQUENCIES": [2.00e9,2.33e9,2.67e9,3.00e9,],
        "VOLTAGES": [0.85,1.00,1.17,1.35,],
    },

    

    #{  # SERVER 2 - Dell PowerEdge R7725 - AMD EPYC 9965, 2 chips, 384 cores total
    #    "MAX_POWER": 861,
    #    "IDLE_POWER": 138,
    #    "MAX_CAPACITY": 40017802,  # SPECpower_ssj2008 ssj_ops @ 100%
    #    "BOOT_TIME": 20,
    #    "SPEC_POWER_CURVE": [
    #        (0, 138),
    #        (10, 297),
    #        (20, 367),
    #        (30, 438),
    #        (40, 515),
    #        (50, 593),
    #        (60, 661),
    #        (70, 710),
    #        (80, 771),
    #        (90, 812),
    #        (100, 861),
    #    ],
    #    "ACTIVITY_FACTOR": 1.0,
    #    "CAPACITANCE": None,
    #},

    # {  # SERVER 3 - Colfax International CX2266-N2 - AMD Opteron 2216HE, 2 chips, 4 cores total
       #  "MAX_POWER": 276,
       #  "IDLE_POWER": 164,
       #  "MAX_CAPACITY": 95853,  # SPECpower_ssj2008 ssj_ops @ 100%
       #  "BOOT_TIME": 20,
       #  "SPEC_POWER_CURVE": [
       #      (0, 164),
       #      (10, 204),
       #      (20, 225),
       #      (30, 234),
       #      (40, 242),
       #      (50, 248),
       #      (60, 254),
       #      (70, 260),
       #      (80, 267),
       #      (90, 272),
       #      (100, 276),
       #   ],
       # "ACTIVITY_FACTOR": 1.0,
       #  "CAPACITANCE": None,
     # },

    #{ #SERVER 4 - ASUSTeK RS720A-E13-RS8U - AMD EPYC 9965, 2 chips, 384 cores total
    #    "MAX_POWER": 800,
    #    "IDLE_POWER": 150,
    #    "MAX_CAPACITY": 39140079,  # SPECpower_ssj2008 ssj_ops @ 100%
    #    "BOOT_TIME": 20,
    #    "SPEC_POWER_CURVE": [
    #        (0, 150),
    #        (10, 265),
    #        (20, 315),
    #        (30, 362),
    #        (40, 413),
    #        (50, 462),
    #        (60, 504),
    #        (70, 546),
    #        (80, 589),
    #        (90, 680),
    #        (100, 800),
    #    ],
    #    "ACTIVITY_FACTOR": 1.0,
    #    "CAPACITANCE": None,
    #},


    # {  # SERVER 5 - HPE ProLiant ML350 Gen11 - Intel Xeon Platinum 8592+, 2 chips, 128 cores total
    #     "MAX_POWER": 629,
    #     "IDLE_POWER": 232,
    #     "MAX_CAPACITY": 13338557,  # SPECpower_ssj2008 ssj_ops @ 100%
    #     "SPEC_POWER_CURVE": [
    #         (0, 232),
    #         (10, 270),
    #         (20, 309),
    #         (30, 346),
    #         (40, 382),
    #         (50, 421),
    #         (60, 462),
    #         (70, 504),
    #         (80, 553),
    #         (90, 595),
    #         (100, 629),
    #     ],
    #    "ACTIVITY_FACTOR": 1.0,
    #     "CAPACITANCE": None,
    # },
]


"""SERVICES"""
CAR_SERVICES = ["connected_vehicles"]
PEDESTRIAN_SERVICES = ["augmented_reality", "virtual_reality"]
STATIONARY_SERVICES = ["video_analysis"]

SERVICES = [
    # name, workload(ops/request), lambda(req/s), vin(bytes), vout(bytes), delay_budget(seconds)

    Service("connected_vehicles", 760, 10, 100, 50, 5e-3),
    Service("augmented_reality", 2700, 0.5, 100, 50, 15e-3),
    Service("video_analysis", 1620, 6, 100, 50, 30e-3),
]


# -------------------ENERGY MODELS---------------------------
#------------------------------------------------------------
# linear model: P = P_idle + (P_max - P_idle) * utilization
# power law model: P = P_idle + (P_max - P_idle) * utilization ** alpha
# empirical model: P = P_idle + (P_max - P_idle) * (2 * utilization - utilization ** alpha) --> (alpha == r ) 
# frequency model: P = P_idle + A * C * V^2 * f -- > DSFV
# spec_linear_interpolation: P = f(utilization) =  --> f is a piecewise linear function defined by the SPECpower_ssj2008 measurements at different utilization levels

# Alpha i s a parameter of calibration for the power law and empirical models.                                                     
# modelos posibles: "linear", "powerlaw"(pasarle alpha), "empirical"(pasarle alpha), "frequency" , "spec_linear_interpolation" 
SERVER_ENERGY_MODEL = "powerlaw"
ALPHA = 0.7
