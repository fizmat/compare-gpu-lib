import numpy as np
from solvers import *

floats = [np.float32, np.float64]
ints = [np.int32, np.float32, np.float64]

def gen_all_types():
  for p in floats:
    for v in floats:
      for t in floats:
        for m in ints:
          for c in ints:
              yield {"pos_t": p, "vel_t": v, "time_t": t, "mass_t": m, "charge_t": c}

def gen_sane_types():
  for f in floats:
    for i in ints:
        yield {"pos_t": f, "vel_t": f, "time_t": f, "mass_t": i, "charge_t": i}

def gen_one_types():
  f = np.float32
  yield {"pos_t": f, "vel_t": f, "time_t": f, "mass_t": f, "charge_t": f}
        
def gen_sane_params():
  for form in CoordFormat:
    for scalar in True, False:
      for compound in True, False:
        yield locals()

def gen_one_params():
  form = CoordFormat.nx
  scalar = False
  compound = True
  yield locals()               
  
sane_tpbs=(32, 64, 128, 256)
one_tpbs=(64,)
many_tbps=(8, 16, 32, 48, 64, 128, 192, 256, 384, 512)
        
