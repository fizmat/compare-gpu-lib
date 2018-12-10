#!/usr/bin/env python
# coding: utf-8

from install import *
from solvers import *
from params import *
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rayleigh, norm, kstest

def plot_maxwell(vel, label=None, draw=True):
  speed = (vel*vel).sum(1)**0.5
  loc, scale = rayleigh.fit(speed, floc=0)
  dist = rayleigh(scale=scale)
  if draw:
    plt.hist(speed, 20, normed=True)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 1000)
    plt.plot(x, dist.pdf(x), label=label)
    if label:
      plt.legend()
  return kstest(speed, dist.cdf)[0]

def plot_maxwell_x(vel, label=None, draw=True):
  loc, scale = norm.fit(vel[:, 0], floc=0)
  dist = norm(scale=scale)
  if draw:
    plt.hist(vel[:, 0], 20, normed=True)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 1000)
    plt.plot(x, dist.pdf(x), label=label)
    if label:
      plt.legend()
  return kstest(vel[:, 0], dist.cdf)[0]

def plot_particles(pos, vel):
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1])

def multi_particles(f, title=None, n=None, width=4.5, height=4):
  if n is None:
    n = min(5, len(f.data))
  fig = plt.figure(figsize=(width*n, height))
  if title:
    fig.suptitle(title)
  for i in range(n):
    plt.subplot(1, n, i+1)
    j = math.floor(i*(len(f.data)-1)/(n-1))
    plt.title(f"t = {f.time[j]:.1f}")
    r = f.data[j]
    plot_particles(r.pos, r.vel)

def multi_maxwell(f, title=None, n=None, draw=True, width=20, height=2):
  if n is None:
    n = len(f.data)
  if draw:
    fig = plt.figure(figsize=(width, height*n))
    if title:
      fig.suptitle(title)
  max_vel = max((r.vel*r.vel).sum(1).max() for r in f.data)**0.5
  max_x = max((np.abs(r.vel[:,0])).max() for r in f.data)
  fits = []
  for i in range(n):
    j = i*(len(f.data)-1)/(n-1)
    r = f.data[j]
    if draw:
      plt.subplot(n, 2, 2*i+1)
      plt.xlim(0, max_vel)
    f1 = plot_maxwell(r.vel, f"t = {f.time[j]:.1f}", draw)
    if draw:
      plt.subplot(n, 2, 2*i+2)
      plt.xlim(-max_x, max_x)
    f2 = plot_maxwell_x(r.vel, f"t = {f.time[j]:.1f}", draw)
    fits.append({"t": f.time[j], "speed_stat":f1, "xvel_stat":f2})
  return pd.DataFrame.from_records(fits, index='t')

def run_simulation(generator, solver, compound_step=True, scale=16,
         time_steps=500, save_step=100):
  dt, n, size, data = generator.gen_on_scale(scale)
  frame = pd.DataFrame(solver.simulate(dt, n, size, time_steps, save_step, compound_step, data), columns=["t", "time", "perf", "data"])
  return frame

def plot_and_fit(frame, title="", draw_particles=True, draw_maxwell=True,
         particles_width=2.5, particles_height=2,
         maxwell_width=10, maxwell_height=1):
  if draw_particles:
    multi_particles(frame, None, width=particles_width, height=particles_height)
  fits = multi_maxwell(frame, title, draw=draw_maxwell, width=maxwell_width, height=maxwell_height)
  return fits


# In[4]:


import sys

def check(generator, solver, compound, **sim_kwargs):
  result = run_simulation(generator, solver, compound, **sim_kwargs)
  fits = plot_and_fit(result, draw_maxwell=False, draw_particles=False)
  fits = pd.DataFrame( {"initial":fits.iloc[0], "final":fits.iloc[-1]}, columns = ("initial","final"))
  fits["improvement"] = fits.initial/fits.final
  return fits, result

def check_fits(SolverClass, tpbs=one_tpbs, types=list(gen_sane_types()), params=list(gen_sane_params()), Generator=TaskGenerator, init_kwargs={}, **sim_kwargs):
  for t in types:
    for p in params:
      for tpb in (1,) if SolverClass.no_blocks else tpbs:
        solver = SolverClass(types=t, coord_format=p['form'], threadsperblock=tpb, scalar=p['scalar'], **init_kwargs)
        result = {k: v.__name__ for k, v in t.items()}
        result.update(p)
        result["tpb"] = tpb
        result["form"] = p["form"].name
        sys.stdout.write("\r" + str(result))
        fits, data = check(Generator(scalar=p['scalar'], **t), solver, p['compound'], **sim_kwargs)
        result["fit_speed"] = fits.final.speed_stat
        result["fit_xvel"] = fits.final.xvel_stat
        result["data"] = data
        yield result

def pick_outliers(checks):
  outliers = checks[np.logical_or(checks.fit_speed==checks.fit_speed.max(), checks.fit_xvel==checks.fit_xvel.max())]
  for d in outliers.data:
    multi_particles(d)
  return outliers

def test_fits(SolverClass, gen_types=gen_sane_types, **sim_kwargs):
  checks = pd.DataFrame.from_records(check_fits(SolverClass, **sim_kwargs))
  for k, c in checks[pd.isnull(checks.fit_speed)].iterrows():
    multi_particles(c['data'], title=str(c.drop('data')))
  for k, c in checks[checks.fit_speed>0.5].iterrows():
    multi_particles(c['data'], title=str(c.drop('data')))
  checks.fit_speed.hist()
  return pick_outliers(checks)
