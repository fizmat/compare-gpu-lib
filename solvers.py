#!/usr/bin/env python
# coding: utf-8

from install import *

# ## Generate parameters and data

from collections import namedtuple
Data = namedtuple('Data', 'charges masses pos vel')
SplitData = namedtuple('SplitData', 'charges masses p0 p1 v0 v1')

import numpy as np
import math

def gen_pos(n, size):
  steps = math.ceil(math.sqrt(n))
  full_grid = np.mgrid[0:steps, 0:steps].reshape(2, steps*steps).transpose()
  n_grid_points = full_grid[:n].copy()
  return size/steps * (n_grid_points+0.5)

def gen_vel(n, max_vel):
  return (np.random.rand(n, 2) - 0.5)*max_vel

class TaskGenerator:
  def __init__(self, pos_t=np.float32, vel_t=np.float32, time_t=np.float32,
               mass_t=np.float32, charge_t=np.float32, scalar=False,
               dist=10, max_vel=1, mass=2, charge=1, dt = 0.1):
    self.dist = pos_t(dist)
    self.max_vel = vel_t(max_vel)
    self.mass = mass_t(mass)
    self.charge = charge_t(charge)
    self.dt = time_t(dt)
    self.scalar = scalar

  def __repr__(self):
    typed = lambda x: f"{type(x)}({x})"
    params = ', '.join(f'{s}={typed(getattr(self, s))}' for s in ('dist', 'max_vel', 'mass', 'charge', 'dt'))
    return f"{self.__class__}({params}, {'scalar' if self.scalar else 'vector'})"

  def __str__(self):
    typed = lambda x: f"{type(x).__name__}({x:.3f})"
    params = ', '.join(f'{s}={typed(getattr(self, s))}' for s in ('dist', 'max_vel', 'mass', 'charge', 'dt'))
    return f"{self.__class__.__name__}({params}, {'scalar' if self.scalar else 'vector'})"

  @staticmethod
  def gen_data(n, size, max_vel, charge, mass, scalar):
    pos = gen_pos(n, size).astype(type(size))
    vel = gen_vel(n, max_vel).astype(type(max_vel))
    charges = charge if scalar else np.full(n, charge)
    masses = mass if scalar else np.full(n, mass)
    return Data(charges, masses, pos, vel)

  def gen_on_scale(self, scale):
    n = scale*scale
    size = type(self.dist)(self.dist * scale)
    return self.dt, n, size, self.gen_data(n, size, self.max_vel, self.charge, self.mass, self.scalar)


# ## Abstract solver class

from time import perf_counter
from enum import Enum, auto
from jinja2 import Template

class CoordFormat(Enum):
  nx = auto()
  xn = auto()
  split = auto()

f=np.float32

class Solver:
  no_blocks = False
  perf_good_batch_crits = (24, 64)
  tpl = Template("")

  def __init__(self, coord_format=CoordFormat.xn, scalar=False, threadsperblock=64, types={}):
    self.format = coord_format
    self.scalar = scalar
    self.threadsperblock = threadsperblock
    f = np.float32
    self.types = {s: f for s in ("pos_t", "vel_t", "time_t", "charge_t", "mass_t")}
    self.types.update(types)
    self.source = self.tpl.render(format=coord_format.name, scalar=scalar,
                                  **{k: v.__name__ for k, v in self.types.items()})

  def __repr__(self):
    return f"{self.__class__.__name__}({self.format.name}, {self.threadsperblock})"

  def _to_device(self, host_array):
    return host_array

  def _to_host(self, dev_array, host_array):
    return dev_array

  def call(self, foo, *args, **kwargs):
    return foo(*args, **kwargs)

  @staticmethod
  def update(self, *args, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def update_v(self, *args, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def update_p(self, *args, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def rebound(self, *args, **kwargs):
    raise NotImplementedError()

  def to_device(self, host_value):
    return self._to_device(host_value) if isinstance(host_value, np.ndarray) else host_value

  def to_host(self, dev_value, host_value):
    return self._to_host(dev_value, host_value) if isinstance(host_value, np.ndarray) else dev_value

  def to_device_all(self, host_data):
    return type(host_data)(*(self.to_device(x) for x in host_data))

  def to_host_all(self, dev_data, host_data):
    return type(host_data)(*(self.to_host(a, b) for a, b in zip(dev_data, host_data)))

  def to_device_layout(self, data):
    if self.format == CoordFormat.nx:
      return Data(*data)
    elif self.format == CoordFormat.xn:
      return Data(data.charges, data.masses, data.pos.transpose().copy(),
                  data.vel.transpose().copy())
    elif self.format == CoordFormat.split:
      return SplitData(data.charges, data.masses,
                       data.pos[:, 0].copy(), data.pos[:, 1].copy(),
                       data.vel[:, 0].copy(), data.vel[:, 1].copy())

  def to_host_layout(self, data):
    if self.format == CoordFormat.nx:
      return Data(*data)
    elif self.format == CoordFormat.xn:
      return Data(data.charges, data.masses, np.copy(data.pos.transpose()),
                  np.copy(data.vel.transpose()))
    elif self.format == CoordFormat.split:
      return Data(data.charges, data.masses,
                  np.stack((data.p0, data.p1), axis=1),
                  np.stack((data.v0, data.v1), axis=1))

  def step(self, dt, size, n, dd, compound_step):
    if self.format == CoordFormat.split:
      if compound_step:
        self.call(self.update_v, dd.p0, dd.p1, dd.v0, dd.v1, dt, dd.charges, dd.masses, n)
        self.call(self.update_p, dd.p0, dd.p1, dd.v0, dd.v1, dt, n)
        self.call(self.rebound, dd.p0, dd.p1, dd.v0, dd.v1, size, n)
      else:
        self.call(self.update, dd.p0, dd.p1, dd.v0, dd.v1, dt, size, dd.charges, dd.masses, n)
    else:
      if compound_step:
        self.call(self.update_v, dd.pos, dd.vel, dt, dd.charges, dd.masses, n)
        self.call(self.update_p, dd.pos, dd.vel, dt, n)
        self.call(self.rebound, dd.pos, dd.vel, size, n)
      else:
        self.call(self.update, dd.pos, dd.vel, dt, size, dd.charges, dd.masses, n)

  def simulate(self, dt, n, size, time_steps, save_step, compound_step, data):
    self.blockspergrid = math.ceil(n / self.threadsperblock)
    batches = [save_step]*(time_steps//save_step) + [time_steps%save_step]
    ti = 0
    host_data = self.to_device_layout(data)
    dev_data = self.to_device_all(host_data)
    host_data = self.to_host_all(dev_data, host_data)
    yield (ti, ti*dt, perf_counter(), self.to_host_layout(host_data))
    for batch_size in batches:
      if batch_size==0: # skip last (remainder) batch if it's not needed
        continue
      for i in range(batch_size):
        self.step(dt, size, np.int32(n), dev_data, compound_step)
      host_data = self.to_host_all(dev_data, host_data)
      ti += batch_size;
      yield (ti, ti*dt, perf_counter(), self.to_host_layout(host_data))


# # Numba

# In[17]:


class NumbaSolver(Solver):
  tpl = Template("""
from numba import cuda
from math import sqrt
import numpy as np

{% set i_thread %}
  i = cuda.grid(1)
  if i < n:
{% endset -%}

{%- if format == "split" %}
  {% set xi, yi, vxi, vyi = 'x[i]', 'y[i]', 'vx[i]', 'vy[i]' %}
  {% set xj, yj = 'x[j]', 'y[j]' %}
  {% set pos = 'x, y'%}
  {% set vel = 'vx, vy' %}
{% elif format == "nx" %}
  {% set xi, yi, vxi, vyi = 'pos[i, 0]', 'pos[i, 1]', 'vel[i, 0]', 'vel[i, 1]' %}
  {% set xj, yj = 'pos[j, 0]', 'pos[j, 1]' %}
  {% set pos, vel = 'pos', 'vel' %}
{% elif format == "xn" %}
  {% set xi, yi, vxi, vyi = 'pos[0, i]', 'pos[1, i]', 'vel[0, i]', 'vel[1, i]' %}
  {% set xj, yj = 'pos[0, j]', 'pos[1, j]' %}
  {% set pos, vel = 'pos', 'vel' %}
{% endif -%}

{%- set update_v %}
    epsilon = np.{{pos_t}}(0.0001);
    f0 = 0
    f1 = 0
    for j in range(n):
      if i != j:
        diff0 = {{xj}} - {{xi}}
        diff1 = {{yj}} - {{yi}}
        dist = sqrt( diff0*diff0 + diff1*diff1 )
        diff0 = diff0 / dist
        diff1 = diff1 / dist
        dist = max(epsilon, dist)
    {%- if scalar %}
        f0 += diff0/(dist*dist)
        f1 += diff1/(dist*dist)
    {%- else %}
        f0 += diff0 * q[j]/(dist*dist)
        f1 += diff1 * q[j]/(dist*dist)
    {%- endif %}
{%- if scalar %}
    {{vxi}} -= q * q * m * dt * f0
    {{vyi}} -= q * q * m * dt * f1
{%- else %}
    {{vxi}} -= q[i] * m[i] * dt * f0
    {{vyi}} -= q[i] * m[i] * dt * f1
{%- endif %}

{% endset -%}

{%- set update_p %}
    {{xi}} += {{vxi}} * dt
    {{yi}} += {{vyi}} * dt
{% endset -%}

{%- set rebound %}
    if {{xi}} < 0:
      {{xi}} = - {{xi}}
      {{vxi}} = - {{vxi}}
    elif {{xi}} > size:
      {{xi}} = 2*size - {{xi}}
      {{vxi}} = - {{vxi}}
    if {{yi}} < 0:
      {{yi}} = - {{yi}}
      {{vyi}} = - {{vyi}}
    elif {{yi}} > size:
      {{yi}} = 2*size - {{yi}}
      {{vyi}} = - {{vyi}}
{% endset -%}

@cuda.jit
def update_v({{pos}}, {{vel}}, dt, q, m, n):
{{ i_thread}}
{{ update_v }}

@cuda.jit
def update_p({{pos}}, {{vel}}, dt, n):
{{ i_thread }}
{{ update_p }}

@cuda.jit
def rebound({{pos}}, {{vel}}, size, n):
{{ i_thread }}
{{ rebound }}

@cuda.jit
def update({{pos}}, {{vel}}, dt, size, q, m, n):
{{ i_thread }}
{{ update_v }}
{{ update_p }}
{{ rebound }}
""")

  def __init__(self, coord_format=CoordFormat.nx, scalar=False, threadsperblock=64, types={}):
    super().__init__(coord_format, scalar, threadsperblock, types)
    context = {}
    exec(self.source, context)
    self.update_v = context['update_v']
    self.update_p = context['update_p']
    self.rebound = context['rebound']
    self.update = context['update']

  def _to_device(self, host_array):
    return ncuda.to_device(host_array)

  def _to_host(self, dev_array, host_array):
    return dev_array.copy_to_host()

  def call(self, foo, *args, **kwargs):
    return foo[self.blockspergrid, self.threadsperblock](*args, **kwargs)


# # CUDA

# In[18]:


class CudaSolver(Solver):
  tpl = Template("""
typedef float float32;
typedef double float64;
typedef int int32;

{% set i_thread -%}
  int i = threadIdx.x + blockIdx.x*blockDim.x;
{% endset -%}

{%- if format == "split" %}
  {% set xi, yi, vxi, vyi = 'x[i]', 'y[i]', 'vx[i]', 'vy[i]' %}
  {% set xj, yj = 'x[j]', 'y[j]' %}
  {% set pos = pos_t + ' *x, ' + pos_t + ' *y' %}
  {% set vel = vel_t + ' *vx, ' + vel_t + ' *vy' %}
{% elif format == "nx" %}
  {% set xi, yi, vxi, vyi = 'pos[2*i]', 'pos[2*i+1]', 'vel[2*i]', 'vel[2*i+1]' %}
  {% set xj, yj = 'pos[2*j]', 'pos[2*j+1]' %}
  {% set pos, vel = pos_t + ' *pos', vel_t + ' *vel' %}
{% elif format == "xn" %}
  {% set xi, yi, vxi, vyi = 'pos[i]', 'pos[n+i]', 'vel[i]', 'vel[n+i]' %}
  {% set xj, yj = 'pos[j]', 'pos[n+j]' %}
  {% set pos, vel = pos_t + ' *pos', vel_t + ' *vel' %}
{% endif -%}

{%- set update_v %}
  {{pos_t}} epsilon = 0.0001;
  {{pos_t}} diff0;
  {{pos_t}} diff1;
  {{pos_t}} dist;
  {{pos_t}} f0 = 0;
  {{pos_t}} f1 = 0;
  for(int j = 0; j<n; j++){
    if(i != j){
      diff0 = {{xj}} - {{xi}};
      diff1 = {{yj}} - {{yi}};
      dist = sqrt( diff0*diff0 + diff1*diff1 );
      diff0 = diff0 / dist;
      diff1 = diff1 / dist;
      dist = max(epsilon, dist);
    {%- if scalar %}
      f0 += diff0/(dist*dist);
      f1 += diff1/(dist*dist);
    {%- else %}
      f0 += diff0 * q[j]/(dist*dist);
      f1 += diff1 * q[j]/(dist*dist);
    {%- endif %}
    }
  }
{%- if scalar %}
    {{vxi}} -= q * q * m * dt * f0;
    {{vyi}} -= q * q * m * dt * f1;
{%- else %}
    {{vxi}} -= q[i] * m[i] * dt * f0;
    {{vyi}} -= q[i] * m[i] * dt * f1;
{%- endif %}
{% endset -%}

{%- set update_p %}
  {{xi}} += {{vxi}} * dt;
  {{yi}} += {{vyi}} * dt;
{% endset -%}

{%- set rebound %}
  if({{xi}} < 0){
    {{xi}} = - {{xi}};
    {{vxi}} = - {{vxi}};
  } else if ({{xi}} > size) {
    {{xi}} = 2*size - {{xi}};
    {{vxi}} = - {{vxi}};
  }
  if({{yi}} < 0){
    {{yi}} = - {{yi}};
    {{vyi}} = - {{vyi}};
  } else if ({{yi}} > size) {
    {{yi}} = 2*size - {{yi}};
    {{vyi}} = - {{vyi}};
  }
{% endset -%}

__global__ void update_v({{pos}}, {{vel}}, {{time_t}} dt,
                         {{charge_t}} {% if not scalar %}*{% endif %}q,
                         {{mass_t}} {% if not scalar %}*{% endif %}m, int n){
  {{ i_thread}}
  if(i < n){
    {{ update_v }}
  }
}

__global__ void update_p({{pos}}, {{vel}}, {{time_t}} dt, int n){
  {{ i_thread }}
  if(i < n){
    {{ update_p }}
  }
}

__global__ void rebound({{pos}}, {{vel}}, {{pos_t}} size, int n){
  {{ i_thread }}
  if(i < n){
    {{ rebound }}
  }
}

__global__ void update({{pos}}, {{vel}}, {{time_t}} dt, {{pos_t}} size,
                       {{charge_t}} {% if not scalar %}*{% endif %}q,
                       {{mass_t}} {% if not scalar %}*{% endif %}m, int n){
  {{ i_thread }}
  if(i < n){
    {{ update_v }}
    {{ update_p }}
    {{ rebound }}
  }
}

""")

  def _to_device(self, host_array):
    return cuda.to_device(host_array)

  def _to_host(self, dev_array, host_array):
    return cuda.from_device_like(dev_array, host_array)

  def __init__(self, coord_format=CoordFormat.nx, scalar=False, threadsperblock=64, types={}):
    super().__init__(coord_format, scalar, threadsperblock, types)
    mod = SourceModule(self.source)
    self.update_v = mod.get_function("update_v")
    self.update_p = mod.get_function("update_p")
    self.rebound = mod.get_function("rebound")
    self.update = mod.get_function("update")

  def call(self, foo, *args, **kwargs):
    grid = (self.blockspergrid, 1)
    block = (self.threadsperblock, 1, 1)
    return foo(block=block, grid=grid, shared=0, *args, **kwargs)


# # OpenCL

# In[19]:


class OpenCLSolver(Solver):
  tpl = Template("""
typedef float float32;
typedef double float64;
typedef int int32;

{% set i_thread -%}
  int i = get_global_id(0);
{% endset -%}

{%- if format == "split" %}
  {% set xi, yi, vxi, vyi = 'x[i]', 'y[i]', 'vx[i]', 'vy[i]' %}
  {% set xj, yj = 'x[j]', 'y[j]' %}
  {% set pos = '__global ' + pos_t + ' *x, __global ' + pos_t + ' *y' %}
  {% set vel = '__global ' + vel_t + ' *vx, __global ' + vel_t + ' *vy' %}
{% elif format == "nx" %}
  {% set xi, yi, vxi, vyi = 'pos[2*i]', 'pos[2*i+1]', 'vel[2*i]', 'vel[2*i+1]' %}
  {% set xj, yj = 'pos[2*j]', 'pos[2*j+1]' %}
  {% set pos, vel = '__global ' + pos_t + ' *pos', '__global ' + vel_t + ' *vel' %}
{% elif format == "xn" %}
  {% set xi, yi, vxi, vyi = 'pos[i]', 'pos[n+i]', 'vel[i]', 'vel[n+i]' %}
  {% set xj, yj = 'pos[j]', 'pos[n+j]' %}
  {% set pos, vel = '__global ' + pos_t + ' *pos', '__global ' + vel_t + ' *vel' %}
{% endif -%}

{%- set update_v %}
  {{pos_t}} epsilon = 0.0001;
  {{pos_t}} diff0;
  {{pos_t}} diff1;
  {{pos_t}} dist;
  {{pos_t}} f0 = 0;
  {{pos_t}} f1 = 0;
  for(int j = 0; j<n; j++){
    if(i != j){
      diff0 = {{xj}} - {{xi}};
      diff1 = {{yj}} - {{yi}};
      dist = sqrt(diff0*diff0 + diff1*diff1);
      diff0 = diff0 / dist;
      diff1 = diff1 / dist;
      dist = max(epsilon, dist);
    {%- if scalar %}
      f0 += diff0/(dist*dist);
      f1 += diff1/(dist*dist);
    {%- else %}
      f0 += diff0 * q[j]/(dist*dist);
      f1 += diff1 * q[j]/(dist*dist);
    {%- endif %}
    }
  }
{%- if scalar %}
  {{vxi}} -= q * q * m * dt * f0;
  {{vyi}} -= q * q * m * dt * f1;
{%- else %}
  {{vxi}} -= q[i] * m[i] * dt * f0;
  {{vyi}} -= q[i] * m[i] * dt * f1;
{%- endif %}
{% endset -%}

{%- set update_p %}
  {{xi}} += {{vxi}} * dt;
  {{yi}} += {{vyi}} * dt;
{% endset -%}

{%- set rebound %}
  if({{xi}} < 0){
    {{xi}} = - {{xi}};
    {{vxi}} = - {{vxi}};
  } else if ({{xi}} > size) {
    {{xi}} = 2*size - {{xi}};
    {{vxi}} = - {{vxi}};
  }
  if({{yi}} < 0){
    {{yi}} = - {{yi}};
    {{vyi}} = - {{vyi}};
  } else if ({{yi}} > size) {
    {{yi}} = 2*size - {{yi}};
    {{vyi}} = - {{vyi}};
  }
{% endset -%}

__kernel void update_v({{pos}}, {{vel}}, {{time_t}} dt,
                         {% if not scalar %}__global{% endif %} {{charge_t}} {% if not scalar %}*{% endif %}q,
                         {% if not scalar %}__global{% endif %} {{mass_t}} {% if not scalar %}*{% endif %}m, int n){
  {{ i_thread}}
  if(i < n){
    {{ update_v }}
  }
}

__kernel void update_p({{pos}}, {{vel}}, {{time_t}} dt, int n){
  {{ i_thread }}
  if(i < n){
    {{ update_p }}
  }
}

__kernel void rebound({{pos}}, {{vel}}, {{pos_t}} size, int n){
  {{ i_thread }}
  if(i < n){
    {{ rebound }}
  }
}

__kernel void update({{pos}}, {{vel}}, {{time_t}} dt, {{pos_t}} size,
                       {% if not scalar %}__global{% endif %} {{charge_t}} {% if not scalar %}*{% endif %}q,
                       {% if not scalar %}__global{% endif %} {{mass_t}} {% if not scalar %}*{% endif %}m, int n){
  {{ i_thread }}
  if(i < n){
    {{ update_v }}
    {{ update_p }}
    {{ rebound }}
  }
}

""")

  def _to_device(self, host_array):
    return cl_array.to_device(self.queue, host_array)

  def _to_host(self, dev_array, host_array):
    return dev_array.get()

  def __init__(self, coord_format=CoordFormat.nx, scalar=False, threadsperblock=64, types={}, platform=0, CPU=False):
    super().__init__(coord_format, scalar, threadsperblock, types)
    platform = cl.get_platforms()[platform]
    my_gpu_devices = platform.get_devices(device_type=cl.device_type.CPU if CPU else cl.device_type.GPU)
    if not my_gpu_devices:
        print("Warning: no GPU devices found!")
        print("Falling back to platform[0].device[0]")
        device = platform.get_devices()[0]
    else:
        device = my_gpu_devices[0]
    self.ctx = cl.Context([device])
    self.queue = cl.CommandQueue(self.ctx)
    program = cl.Program(self.ctx, self.source).build()
    q_t, m_t = (self.types["charge_t"], self.types["mass_t"]) if scalar else (None, None)
    arr_n = 4 if coord_format == CoordFormat.split else 2
    self.update_v = program.update_v
    self.update_v.set_scalar_arg_dtypes([None]*arr_n + [self.types['time_t'], q_t, m_t, np.int32 ] )
    self.update_p = program.update_p
    self.update_p.set_scalar_arg_dtypes([None]*arr_n + [self.types['time_t'], np.int32])
    self.rebound = program.rebound
    self.rebound.set_scalar_arg_dtypes([None]*arr_n + [self.types['pos_t'], np.int32 ])
    self.update = program.update
    self.update.set_scalar_arg_dtypes([None]*arr_n + [self.types['time_t'], self.types['pos_t'],
                                                      q_t, m_t, np.int32 ])


  def call(self, foo, *args, **kwargs):
    global_work = [self.blockspergrid * self.threadsperblock]
    local_work = [self.threadsperblock]
    args = [a.data if isinstance(a, cl_array.Array) else a for a in args]
    kwargs = {k: a.data if isinstance(a, cl_array.Array) else a for k, a in kwargs.items()}
    return foo(self.queue, global_work, local_work, *args, **kwargs)


# # Numpy

# In[20]:


class NumpySolver(Solver):
  perf_good_batch_crits = (6, 8)
  no_blocks = True
  tpl = Template("""
{% if format == "split" %}
{% set pos = 'x, y'%}
{% set vel = 'vx, vy' %}
{%- set update_p %}
    x += vx*dt
    y += vy*dt
{% endset -%}
{%- set rebound %}
    below_zero = x < 0
    x[below_zero] = -x[below_zero]
    vx[below_zero] = -vx[below_zero]
    above_size = x > size
    x[above_size] = 2*size - x[above_size]
    vx[above_size] = -vx[above_size]
    below_zero = y < 0
    y[below_zero] = -y[below_zero]
    vy[below_zero] = -vy[below_zero]
    above_size = y > size
    y[above_size] = 2*size - y[above_size]
    vy[above_size] = -vy[above_size]
{% endset -%}
{% else -%}
{% set pos, vel = 'p', 'v' %}
{%- set update_p %}
    p += v*dt
{% endset -%}
{%- set rebound %}
    below_zero = p < 0
    p[below_zero] = -p[below_zero]
    v[below_zero] = -v[below_zero]
    above_size = p > size
    p[above_size] = 2*size - p[above_size]
    v[above_size] = -v[above_size]
{% endset -%}
{% endif -%}

{%- if format == "split" %}
{%- set update_v %}
    epsilon = np.{{pos_t}}(0.0001)
    dx, dy = x[np.newaxis] - x[:, np.newaxis], y[np.newaxis] - y[:, np.newaxis] # pairwise distance vectors
    dist = np.sqrt(dx*dx + dy*dy) # absolute distances
    np.fill_diagonal(dist, 1.0)
    dx = dx / dist # normalized distance vectors
    dy = dy / dist # normalized distance vectors
    dist = np.maximum(epsilon, dist)
    {% if scalar -%}
    fx = dx / (dist**2)
    fy = dy / (dist**2)
    vx -= fx.sum(1) * (dt * (q * q / m))
    vy -= fy.sum(1) * (dt * (q * q / m))
    {% else -%}
    fx = q[:, np.newaxis] * q[np.newaxis, :] * dx / (dist**2)
    fy = q[:, np.newaxis] * q[np.newaxis, :] * dy / (dist**2)
    vx -= fx.sum(1) * dt / m
    vy -= fy.sum(1) * dt / m
    {% endif -%}
{% endset -%}
{% elif format == "nx" %}
{%- set update_v %}
    epsilon = np.{{pos_t}}(0.0001)
    diff = p[np.newaxis] - p[:, np.newaxis] # pairwise distance vectors
    dist = np.sqrt((diff*diff).sum(2)) # absolute distances
    np.fill_diagonal(dist, 1.0)
    diff = diff / dist[:, :, np.newaxis] # normalized distance vectors
    dist = np.maximum(epsilon, dist)
    {% if scalar -%}
    f = diff / (dist**2)[:, :, np.newaxis]
    v -= f.sum(1) * (dt * (q * q / m))
    {% else -%}
    f = (q[:, np.newaxis] * q[np.newaxis] / dist**2)[:, :, np.newaxis] * diff
    v -= f.sum(1) * dt / m[:, np.newaxis]
    {% endif -%}
{% endset -%}
{% elif format == "xn" %}
{%- set update_v %}
    epsilon = np.{{pos_t}}(0.0001)
    diff = p[:, np.newaxis] - p[:, :, np.newaxis] # pairwise distance vectors
    dist = np.sqrt((diff*diff).sum(0)) # absolute distances
    np.fill_diagonal(dist, 1.0)
    diff = diff / dist # normalized distance vectors
    dist = np.maximum(epsilon, dist)
    {% if scalar -%}
    f = diff / (dist**2)
    v -= f.sum(2) * (dt * (q*q/m))
    {% else -%}
    f = (q[:, np.newaxis] * q[np.newaxis, :] / dist**2)[np.newaxis] * diff
    v -= f.sum(2) * dt / m
    {% endif -%}
{% endset -%}
{% endif -%}

def update_v({{pos}}, {{vel}}, dt, q, m, n):
{{ update_v }}

def update_p({{pos}}, {{vel}}, dt, n):
{{ update_p }}

def rebound({{pos}}, {{vel}}, size, n):
{{ rebound }}

def update({{pos}}, {{vel}}, dt, size, q, m, n):
{{ update_v }}
{{ update_p }}
{{ rebound }}

""")

  def __init__(self, coord_format=CoordFormat.xn, scalar=False, threadsperblock=64, types={}):
    super().__init__(coord_format, scalar, threadsperblock, types)
    context = {"np": np}
    exec(self.source, context)
    self.update_v = context['update_v']
    self.update_p = context['update_p']
    self.rebound = context['rebound']
    self.update = context['update']


# # Extra CPU solvers

import numpy as np
from numba import jit

## compound split scalar only

@jit(nopython=True, parallel=True)
def update_v(x, y, vx, vy, dt, q, m, n):
  epsilon = np.float32(0.0001)
  for i in range(n):
    dx = x - x[i]
    dy = y - y[i]
    dist = np.sqrt(dx*dx + dy*dy) # absolute distances
    dist[i] = 1.0
    dx = dx / dist # normalized distance
    dy = dy / dist
    dist = np.maximum(epsilon, dist)
    fx = (q*q) * ( dx / (dist**2)).sum()
    fy = (q*q) * ( dy / (dist**2)).sum()
    vx[i] -= fx/m * dt
    vy[i] -= fy/m * dt

@jit(nopython=True, parallel=True)
def update_p(x, y, vx, vy, dt, n):
  x += vx*dt
  y += vy*dt

@jit(nopython=True, parallel=True)
def rebound(x, y, vx, vy, size, n):
  below_zero = x < 0
  x[below_zero] = -x[below_zero]
  vx[below_zero] = -vx[below_zero]
  above_size = x > size
  x[above_size] = 2*size - x[above_size]
  vx[above_size] = -vx[above_size]
  below_zero = y < 0
  y[below_zero] = -y[below_zero]
  vy[below_zero] = -vy[below_zero]
  above_size = y > size
  y[above_size] = 2*size - y[above_size]
  vy[above_size] = -vy[above_size]

class NumbaJitSolver(NumpySolver):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.update_v = update_v
    self.update_p = update_p
    self.rebound = rebound

class PythonSolver(NumpySolver):
  perf_good_batch_crits = (2, 2)
  tpl = Template("""
from math import sqrt
import numpy as np

{% set i_thread %}
  for i in range(n):
{% endset -%}

{%- if format == "split" %}
  {% set xi, yi, vxi, vyi = 'x[i]', 'y[i]', 'vx[i]', 'vy[i]' %}
  {% set xj, yj = 'x[j]', 'y[j]' %}
  {% set pos = 'x, y'%}
  {% set vel = 'vx, vy' %}
{% elif format == "nx" %}
  {% set xi, yi, vxi, vyi = 'pos[i, 0]', 'pos[i, 1]', 'vel[i, 0]', 'vel[i, 1]' %}
  {% set xj, yj = 'pos[j, 0]', 'pos[j, 1]' %}
  {% set pos, vel = 'pos', 'vel' %}
{% elif format == "xn" %}
  {% set xi, yi, vxi, vyi = 'pos[0, i]', 'pos[1, i]', 'vel[0, i]', 'vel[1, i]' %}
  {% set xj, yj = 'pos[0, j]', 'pos[1, j]' %}
  {% set pos, vel = 'pos', 'vel' %}
{% endif -%}

{%- set update_v %}
    epsilon = np.{{pos_t}}(0.0001)
    f0 = f1 = 0
    for j in range(n):
      if i != j:
        diff0 = {{xj}} - {{xi}}
        diff1 = {{yj}} - {{yi}}
        dist = sqrt( diff0*diff0 + diff1*diff1 )
        diff0 = diff0 / dist
        diff1 = diff1 / dist
        dist = max(epsilon, dist)
      {%- if scalar %}
        f0 += diff0/(dist*dist)
        f1 += diff1/(dist*dist)
      {%- else %}
        f0 += diff0 * q[j]/(dist*dist)
        f1 += diff1 * q[j]/(dist*dist)
      {%- endif %}
{%- if scalar %}
    {{vxi}} -= q * q * m * dt * f0
    {{vyi}} -= q * q * m * dt * f1
{%- else %}
    {{vxi}} -= q[i] * m[i] * dt * f0
    {{vyi}} -= q[i] * m[i] * dt * f1
{%- endif %}
{% endset -%}

{%- set update_p %}
    {{xi}} += {{vxi}} * dt
    {{yi}} += {{vyi}} * dt
{% endset -%}

{%- set rebound %}
    if {{xi}} < 0:
      {{xi}} = - {{xi}}
      {{vxi}} = - {{vxi}}
    elif {{xi}} > size:
      {{xi}} = 2*size - {{xi}}
      {{vxi}} = - {{vxi}}
    if {{yi}} < 0:
      {{yi}} = - {{yi}}
      {{vyi}} = - {{vyi}}
    elif {{yi}} > size:
      {{yi}} = 2*size - {{yi}}
      {{vyi}} = - {{vyi}}
{% endset -%}

def update_v({{pos}}, {{vel}}, dt, q, m, n):
{{ i_thread}}
{{ update_v }}

def update_p({{pos}}, {{vel}}, dt, n):
{{ i_thread }}
{{ update_p }}

def rebound({{pos}}, {{vel}}, size, n):
{{ i_thread }}
{{ rebound }}

def update({{pos}}, {{vel}}, dt, size, q, m, n):
{{ i_thread }}
{{ update_v }}
{{ update_p }}
{{ rebound }}
""")
