# instance.py
"""
Generates processing time matrices in the style of Taillard's Flow Shop benchmarks.
Uses the same Lehmer RNG parameters and seed table as the original instances.
"""
import sys
from typing import List

# =============================================================================
# Taillard benchmark seed values (first 120 values – enough for most instances)
# =============================================================================
time_seeds = [
    873654221,379008056,1866992158,216771124,495070989,
    402959317,1369363414,2021925980,573109518,88325120,
    587595453,1401007982,873136276,268827376,1634173168,
    691823909,73807235,1273398721,2065119309,1672900551,
    479340445,268827376,1958948863,918272953,555010963,
    2010851491,1519833303,1748670931,1923497586,1829909967,
    1328042058,200382020,496319842,1203030903,1730708564,
    450926852,1303135678,1273398721,587288402,248421594,
    1958948863,575633267,655816003,1977864101,93805469,
    1803345551,49612559,1899802599,2013025619,578962478,
    1539989115,691823909,655816003,1315102446,1949668355,
    1923497586,1805594913,1861070898,715643788,464843328,
    896678084,1179439976,1122278347,416756875,267829958,
    1835213917,1328833962,1418570761,161033112,304212574,
    1539989115,655816003,960914243,1915696806,2013025619,
    1168140026,1923497586,167698528,1528387973,993794175,
    450926852,1462772409,1021685265,83696007,508154254,
    1861070898,26482542,444956424,2115448041,118254244,
    471503978,1215892992,135346136,1602504050,160037322,
    551454346,519485142,383947510,1968171878,540872513,
    2013025619,475051709,914834335,810642687,1019331795,
    2056065863,1342855162,1325809384,1988803007,765656702,
    1368624604,450181436,1927888393,1759567256,606425239,
    19268348,1298201670,2041736264,379756761,28837162
]

def taillard_get_nb_jobs(inst_id: int) -> int:
    """Returns number of jobs (n) for Taillard instance taXX"""
    if inst_id > 110:
        return 500
    if inst_id > 90:
        return 200
    if inst_id > 60:
        return 100
    if inst_id > 30:
        return 50
    return 20

def taillard_get_nb_machines(inst_id: int) -> int:
    """Returns number of machines (m) for Taillard instance taXX"""
    if inst_id > 110:
        return 20
    if inst_id > 100:
        return 20
    if inst_id > 90:
        return 10
    if inst_id > 80:
        return 20
    if inst_id > 70:
        return 10
    if inst_id > 60:
        return 5
    if inst_id > 50:
        return 20
    if inst_id > 40:
        return 10
    if inst_id > 30:
        return 5
    if inst_id > 20:
        return 20
    if inst_id > 10:
        return 10
    return 5

def lehmer_unif(seed: List[int], low: int, high: int) -> int:
    """
    Lehmer random number generator - same parameters as used in Taillard's original C code
    seed must be a mutable list with one integer (emulates pointer)
    Returns integer in [low, high] inclusive
    """
    m = 2147483647
    a = 16807
    b = 127773
    c = 2836
    k = seed[0] // b
    seed[0] = a * (seed[0] % b) - k * c
    if seed[0] < 0:
        seed[0] += m
    value_0_1 = seed[0] / m
    return low + int(value_0_1 * (high - low + 1))

def generate_taillard_processing_times(inst_id: int) -> List[List[int]]:
    """
    Generate the exact processing time matrix used in Taillard benchmark instance taXX
    Returns:
        List[List[int]]: matrix of shape (m × n) – machines × jobs
    """
    if inst_id < 1 or inst_id > len(time_seeds):
        raise ValueError(f"Instance ID must be between 1 and {len(time_seeds)}")
    n = taillard_get_nb_jobs(inst_id)
    m = taillard_get_nb_machines(inst_id)
    seed = [time_seeds[inst_id - 1]]
    ptm = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ptm[i][j] = lehmer_unif(seed, 1, 99)
    return ptm

def print_taillard_instance(inst_id: int, comment: str = ""):
    """Print instance in a format similar to Taillard benchmark files"""
    n = taillard_get_nb_jobs(inst_id)
    m = taillard_get_nb_machines(inst_id)
    pt = generate_taillard_processing_times(inst_id)
    print(f"Ta{inst_id:02d}")
    if comment:
        print(f"# {comment}")
    print(f"{n} {m}\n")
    for machine in range(m):
        print(" ".join(f"{x:2}" for x in pt[machine]))
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(" python instance.py <inst_id> [comment]")
        sys.exit(1)
    try:
        inst_id = int(sys.argv[1])
        comment = sys.argv[2] if len(sys.argv) > 2 else ""
        print_taillard_instance(inst_id, comment)
    except ValueError:
        print("Error: instance ID must be an integer")
        sys.exit(1)
