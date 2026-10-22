"""Microscopic validation in SUMO of a signal plan from the 48-variable model.

Mapping: model intersection idx(r, c) -> SUMO junction <letter c><digit r>;
row routes (model phase "EW") run along x, column routes (phase "NS") along y.
Each intersection runs a two-phase program: EW green, 3 s yellow, 1 s all-red,
NS green, 3 s yellow, 1 s all-red (cycle = gEW + gNS + 8 s = C), with the EW
green starting at the model offset. Vehicles: through trips only, demand from
the model scenario, 15 min warm-up + 60 min measurement, HBEFA emission model.
"""
import os, sys, subprocess, tempfile, xml.etree.ElementTree as ET
import numpy as np

SUMO_HOME = "/usr/local/lib/python3.13/dist-packages/sumo"
os.environ["SUMO_HOME"] = SUMO_HOME
SUMO = f"{SUMO_HOME}/bin/sumo"
HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "grid.net.xml")
sys.path.insert(0, "/home/claude/tlmowoa")
import problem as P

L = "ABCD"
WARM, MEAS = 900, 3600


def _classify():
    r = ET.parse(NET).getroot()
    xy = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in r.findall("junction")}
    efrom = {e.get("id"): e.get("from") for e in r.findall("edge")}
    links = {}
    for c in r.findall("connection"):
        tl = c.get("tl")
        if tl is None:
            continue
        fx, fy = xy[efrom[c.get("from")]]
        jx, jy = xy[tl]
        horiz = abs(fy - jy) < 1e-6
        links.setdefault(tl, {})[int(c.get("linkIndex"))] = horiz
    return links


LINKS = _classify()


def _state(tl, horiz_sig, vert_sig):
    d = LINKS[tl]
    return "".join(horiz_sig if d[i] else vert_sig for i in range(max(d) + 1))


def write_tls(x, path, offset_sign=1):
    x = np.clip(np.asarray(x, float), P.LB, P.UB)
    Cy, sp, of = x[0::3], x[1::3], x[2::3]
    out = ["<additional>"]
    for r_ in range(P.R):
        for c_ in range(P.C_GRID):
            i = P.idx(r_, c_)
            tl = f"{L[c_]}{r_}"
            eff = Cy[i] - P.LOST
            gNS, gEW = sp[i] * eff, (1 - sp[i]) * eff
            off = offset_sign * of[i] * Cy[i]
            out.append(f'  <tlLogic id="{tl}" type="static" programID="opt" offset="{off:.1f}">')
            for dur, st in ((gEW, _state(tl, "G", "r")), (3, _state(tl, "y", "r")), (1, _state(tl, "r", "r")),
                            (gNS, _state(tl, "r", "G")), (3, _state(tl, "r", "y")), (1, _state(tl, "r", "r"))):
                out.append(f'    <phase duration="{max(dur, 1.0):.1f}" state="{st}"/>')
            out.append("  </tlLogic>")
    out.append("</additional>")
    open(path, "w").write("\n".join(out))


def write_routes(demand, path):
    out = ['<routes>', '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
           'maxSpeed="13.89" emissionClass="HBEFA4/PC_petrol_Euro-4"/>']
    k = 0
    for (ph, seq), q in zip(P.ROUTES, demand):
        names = []
        if ph == "EW":
            r_ = seq[0] // P.C_GRID
            cols = [s % P.C_GRID for s in seq]
            fwd = cols[0] == 0
            a, b = ("left", "right") if fwd else ("right", "left")
            nodes = [f"{a}{r_}"] + [f"{L[c]}{r_}" for c in cols] + [f"{b}{r_}"]
        else:
            c_ = seq[0] % P.C_GRID
            rows = [s // P.C_GRID for s in seq]
            fwd = rows[0] == 0
            a, b = ("bottom", "top") if fwd else ("top", "bottom")
            nodes = [f"{a}{c_}"] + [f"{L[c_]}{r}" for r in rows] + [f"{b}{c_}"]
        edges = " ".join(nodes[j] + nodes[j + 1] for j in range(len(nodes) - 1))
        out.append(f'  <flow id="f{k}" type="car" begin="0" end="{WARM + MEAS}" vehsPerHour="{q:.1f}" '
                   f'departLane="best" departSpeed="max" route="r{k}"/>'.replace(f'route="r{k}"', f'route="r{k}"'))
        out.insert(-1, f'  <route id="r{k}" edges="{edges}"/>')
        k += 1
    out.append("</routes>")
    open(path, "w").write("\n".join(out))


def simulate(x, demand=P.TARGET, seed=0, keep=False):
    d = tempfile.mkdtemp(prefix="sv_")
    write_tls(x, f"{d}/tls.add.xml")
    write_routes(demand, f"{d}/routes.rou.xml")
    cmd = [SUMO, "-n", NET, "-r", f"{d}/routes.rou.xml", "-a", f"{d}/tls.add.xml",
           "--begin", "0", "--end", str(WARM + MEAS + 1800), "--seed", str(seed),
           "--tripinfo-output", f"{d}/trip.xml", "--device.emissions.probability", "1",
           "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300",
           "--step-length", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    tt, fuel, co2, dist, delay = [], [], [], [], []
    for t in ET.parse(f"{d}/trip.xml").getroot().findall("tripinfo"):
        dep = float(t.get("depart"))
        if dep < WARM or dep >= WARM + MEAS:
            continue
        em = t.find("emissions")
        tt.append(float(t.get("duration")) + float(t.get("departDelay")))
        delay.append(float(t.get("timeLoss")) + float(t.get("departDelay")))
        dist.append(float(t.get("routeLength")))
        fuel.append(float(em.get("fuel_abs")))
        co2.append(float(em.get("CO2_abs")))
    dist_km = np.sum(dist) / 1000
    res = {"n_trips": len(tt), "travel_time_min": np.mean(tt) / 60, "delay_s": np.mean(delay),
           "fuel_L100km": np.sum(fuel) / 1e3 / 0.745 / dist_km * 100 if False else None,
           "co2_g_km": np.sum(co2) / 1e3 / dist_km}
    # fuel_abs is reported in mg (SUMO >= 1.14 for HBEFA4: mg); convert via petrol density 745 g/L
    res["fuel_L100km"] = np.sum(fuel) / 1e3 / 745.0 / dist_km * 100
    if not keep:
        subprocess.run(["rm", "-rf", d])
    else:
        res["dir"] = d
    return res
