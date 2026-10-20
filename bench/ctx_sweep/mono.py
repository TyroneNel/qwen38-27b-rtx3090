"""mono.py <script> [args...]: runs a bench script with time.time on the monotonic clock.
Under WSL2 Docker Desktop the VM's wall clock steps back about 2.3 s every 28 s, which corrupts a duration taken
with time.time(). labd_bench.py on main times with time.time().
"""
import os
import runpy
import sys
import time

time.time = time.perf_counter
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
