"""Historical optimization data D_s: one MOWOA run per source demand scenario."""
import sys, pickle, numpy as np
import problem as P
from algorithms import run_woa_family

out = []
for i, (name, dem) in enumerate(P.SOURCES.items()):
    arch, rec, info = run_woa_family(dem, 1000 + i, keep=True)
    st = info["ev"].store
    out.append({"name": name, "demand": dem,
                "X_all": np.vstack([s[0] for s in st]), "F_all": np.vstack([s[1] for s in st]),
                "V_all": np.concatenate([s[2] for s in st]),
                "X_arch": arch.X[arch.V == 0] if (arch.V == 0).any() else arch.X})
    print(name, "evals", len(out[-1]["X_all"]), "feasible arch", int((arch.V == 0).sum()), flush=True)
pickle.dump(out, open(sys.argv[1], "wb"))
