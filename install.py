#!/usr/bin/env python
# coding: utf-8

# In[7]:


import platform, os
dist = platform.dist()[0]
if dist == 'Ubuntu':
  get_ipython().system('apt install -y --no-install-recommends -q nvidia-cuda-toolkit')
  get_ipython().system('apt-get install lshw clinfo')
  
  get_ipython().system('pip install numba')
  os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/lib/nvidia-cuda-toolkit/libdevice"
  os.environ['NUMBAPRO_NVVM'] = "/usr/lib/x86_64-linux-gnu/libnvvm.so"
  
  get_ipython().system('pip install pycuda')
  
  get_ipython().system('pip install pyopencl==2016.1')

elif dist == 'arch':
  get_ipython().system('sudo pacman -S cuda lshw clinfo  --needed --noconfirm')
  cuda_bin = "/opt/cuda/bin/"
  if not cuda_bin in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + cuda_bin
  
  get_ipython().system('pip install numba')
  os.environ['NUMBAPRO_LIBDEVICE'] = "/opt/cuda/nvvm/libdevice"
  os.environ['NUMBAPRO_NVVM'] = "/opt/cuda/nvvm/lib64/libnvvm.so"

  get_ipython().system('sudo pacman -S python-pycuda --needed --noconfirm')
  
  get_ipython().system('sudo pacman -S python-pyopencl --needed --noconfirm')


# In[8]:


get_ipython().system('nvcc --version')
get_ipython().system('nvidia-smi')
get_ipython().system('lshw -numeric -C display')
get_ipython().system('clinfo')


# In[9]:


from numba import cuda as ncuda
from numba import jit


# In[10]:


import pycuda.autoinit
import pycuda.driver as cuda
from pprint import pprint
from pycuda.compiler import SourceModule
(free,total)=cuda.mem_get_info()
print(free, total)
print( "Global memory occupancy: {}% free".format( free * 100 / total ) )

for devicenum in range( cuda.Device.count() ):
  device = cuda.Device( devicenum )
  attrs = device.get_attributes()
  pprint(attrs)


# In[11]:


import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as cl_array

print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
for platform in cl.get_platforms():  # Print each platform on this computer
    print('=' * 60)
    print('Platform - Name:  ' + platform.name)
    print('Platform - Vendor:  ' + platform.vendor)
    print('Platform - Version:  ' + platform.version)
    print('Platform - Profile:  ' + platform.profile)
    for device in platform.get_devices():  # Print each device per-platform
        print('    ' + '-' * 56)
        print('    Device - Name:  ' + device.name)
        print('    Device - Type:  ' + cl.device_type.to_string(device.type))
        print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
        print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
        print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024))
        print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024))
        print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
print('\n')


# In[ ]:




