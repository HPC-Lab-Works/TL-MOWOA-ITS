"""Constrained four-objective signal-timing problem on a 4x4 signalized grid.

Macroscopic evaluation model (vectorised over a population):
  * HCM-style control delay: uniform delay d1 with a progression factor derived
    from offsets, plus incremental (oversaturation) delay d2.
  * Akcelik-style elemental fuel model (cruise + idle + stop penalties);
    CO2 from fuel with a gasoline/diesel fleet mix.
Decision vector (n = 48): for each of the 16 intersections
  [cycle length C in [40,180] s, green split for N-S phase in [0.15,0.85],
   offset fraction in [0,1)].
Objectives (all minimised):
  f1 average travel time per vehicle (min)
  f2 flow-weighted mean degree of saturation (congestion index)
  f3 CO2 emissions (g/km)
  f4 fuel consumption (L/100 km)
Constraints g_j <= 0:
  g1 lane back-of-queue <= Q_MAX (link storage)
  g2 cycle length within [60,150] s and every effective green >= 10 s
  g3 degree of saturation <= 1 on every approach
"""
import numpy as np

R = C_GRID = 4
NI = R * C_GRID
NVAR = 3 * NI
LB = np.tile([40.0, 0.15, 0.0], NI)
UB = np.tile([180.0, 0.85, 1.0], NI)

S_LANE = 1800.0      # saturation flow veh/h/lane
LANES = 2
S_APP = S_LANE * LANES
LOST = 8.0           # lost time per cycle (s), two phases
LINK_M = 400.0
V_FREE = 50 / 3.6
TT_LINK = LINK_M / V_FREE   # 28.8 s
N_LINKS = R + 1             # links traversed per through trip
Q_MAX = 30.0                # veh per lane storage (~225 m)
C_MIN, C_MAX, G_MIN = 60.0, 150.0, 10.0
T_ANAL, K_INC = 0.25, 0.5   # analysis period (h), incremental delay k
P_RANDOM = 0.3              # share of non-platooned arrivals

# fuel model coefficients (Akcelik elemental model, light vehicles)
F_CRUISE = 0.075     # L/km at 50 km/h
F_IDLE = 1.2 / 3600  # L/s
F_STOP = 0.013       # L per full stop from 50 km/h
CO2_PER_L = 0.85 * 2310 + 0.15 * 2680   # g/L, fleet mix


def idx(r, c):
    return r * C_GRID + c


def decode(X):
    X = np.atleast_2d(X)
    Cy = X[:, 0::3]
    sp = X[:, 1::3]
    off = X[:, 2::3] * Cy
    eff = Cy - LOST
    gNS = sp * eff
    gEW = (1 - sp) * eff
    # phase EW starts at offset, NS starts after EW green + half lost time
    stEW = off
    stNS = off + gEW + LOST / 2
    return Cy, gNS, gEW, stNS, stEW


def _routes():
    """Through routes: each row in two directions (EW phase), each column in
    two directions (NS phase). Returns list of (phase, ordered intersections)."""
    routes = []
    for r in range(R):
        seq = [idx(r, c) for c in range(C_GRID)]
        routes.append(("EW", seq))
        routes.append(("EW", seq[::-1]))
    for c in range(C_GRID):
        seq = [idx(r, c) for r in range(R)]
        routes.append(("NS", seq))
        routes.append(("NS", seq[::-1]))
    return routes


ROUTES = _routes()
N_ROUTES = len(ROUTES)  # 16


def evaluate(X, demand):
    """X: (P, 48) array; demand: (16,) entry flows veh/h per route.
    Returns F (P,4) and G (P,3)."""
    X = np.clip(np.atleast_2d(X), LB, UB)
    P = X.shape[0]
    Cy, gNS, gEW, stNS, stEW = decode(X)
    tot_veh = 0.0
    tot_time = np.zeros(P)
    tot_fuel = np.zeros(P)
    tot_vkm = 0.0
    x_w = np.zeros(P)
    q_w = 0.0
    qmax = np.zeros(P)
    xmax = np.zeros(P)
    for (ph, seq), q0 in zip(ROUTES, demand):
        g_all = gNS if ph == "NS" else gEW
        st_all = stNS if ph == "NS" else stEW
        q = np.full(P, q0)
        delay = np.zeros(P)
        stops = np.zeros(P)
        prev = None
        for k, i in enumerate(seq):
            C = Cy[:, i]
            g = g_all[:, i]
            u = g / C
            cap = S_APP * u
            Xd = q / cap
            # progression: platoon released at upstream green start
            if prev is None:
                Pg = u
            else:
                Cu, gu, su = Cy[:, prev], g_all[:, prev], st_all[:, prev]
                w = np.exp(-((C - Cu) / 3.0) ** 2)          # common-cycle coupling
                arr = (su + TT_LINK - st_all[:, i]) % C       # platoon head, rel. to green start
                plat = np.minimum(gu, C)
                # overlap of [arr, arr+plat] with green [0, g] on a circle of length C
                ov = np.zeros(P)
                for shift in (-1.0, 0.0, 1.0):
                    a = arr + shift * C
                    ov += np.clip(np.minimum(a + plat, g) - np.maximum(a, 0.0), 0, None)
                Pc = np.clip(ov / plat, 0, 1)
                Pp = (1 - P_RANDOM) * Pc + P_RANDOM * u
                Pg = w * Pp + (1 - w) * u
            PF = np.clip((1 - Pg) / np.maximum(1 - u, 1e-6), 0.05, 2.5)
            d1 = 0.5 * C * (1 - u) ** 2 / (1 - np.minimum(1.0, Xd) * u)
            cT = cap * T_ANAL
            d2 = 900 * T_ANAL * ((Xd - 1) + np.sqrt((Xd - 1) ** 2 + 8 * K_INC * Xd / cT))
            d = d1 * PF + d2
            y = q / S_APP
            h = 0.9 * (1 - Pg) / np.maximum(1 - y, 0.05) + np.maximum(Xd - 1, 0) * 2.0
            h = np.minimum(h, 2.5)
            # back of queue per lane (red-time arrivals scaled by PF + overflow)
            Q1 = (q / 3600) * C * (1 - u) * PF / np.maximum(1 - np.minimum(1, Xd) * u, 1e-3) / LANES
            Q2 = 0.25 * cT * ((Xd - 1) + np.sqrt((Xd - 1) ** 2 + 8 * K_INC * Xd / cT)) / LANES
            qmax = np.maximum(qmax, Q1 + Q2)
            xmax = np.maximum(xmax, Xd)
            x_w += Xd * q
            q_w += q0
            delay += d
            stops += h
            q = np.minimum(q, cap)   # downstream flow metered by capacity
            prev = i
        tt = N_LINKS * TT_LINK + delay
        dist_km = N_LINKS * LINK_M / 1000
        fuel = F_CRUISE * dist_km + F_IDLE * delay + F_STOP * stops
        tot_veh += q0
        tot_time += q0 * tt
        tot_fuel += q0 * fuel
        tot_vkm += q0 * dist_km
    f1 = tot_time / tot_veh / 60.0
    f2 = x_w / q_w
    f4 = tot_fuel / tot_vkm * 100.0
    f3 = tot_fuel * CO2_PER_L / tot_vkm
    F = np.stack([f1, f2, f3, f4], axis=1)
    gminv = np.maximum(0, G_MIN - np.minimum(gNS, gEW)).sum(1) / G_MIN
    cyv = (np.maximum(0, C_MIN - Cy) + np.maximum(0, Cy - C_MAX)).sum(1) / 10.0
    G = np.stack([qmax / Q_MAX - 1.0, cyv + gminv, xmax - 1.0], axis=1)
    return F, G


def violation(G):
    return np.maximum(G, 0).sum(1)


# ---------------- demand scenarios (veh/h per route, 2-lane approaches) ----------------
def _pattern(row_main, row_opp, col_main, col_opp, row_scale, col_scale):
    d = []
    for r in range(R):
        d += [row_main * row_scale[r], row_opp * row_scale[r]]
    for c in range(C_GRID):
        d += [col_main * col_scale[c], col_opp * col_scale[c]]
    return np.array(d)


CENTRE = [0.9, 1.15, 1.15, 0.9]
FLAT = [1.0, 1.0, 1.0, 1.0]
TARGET = _pattern(1250, 700, 1150, 650, CENTRE, CENTRE)          # heavy AM peak
SOURCES = {
    "offpeak": _pattern(650, 550, 600, 500, FLAT, FLAT),
    "pm_peak": _pattern(700, 1150, 650, 1050, CENTRE, CENTRE),   # reversed direction
    "shoulder": _pattern(1000, 600, 900, 550, FLAT, FLAT),
    "am_prev_year": _pattern(1100, 650, 1000, 600, CENTRE, [1.0, 1.1, 1.1, 1.0]),
}


def demand_features(dem):
    return np.asarray(dem) / 1500.0
