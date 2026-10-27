"""Attribute the run time of allel2 analyses to Cython kernels, NumPy and I/O.

cProfile cannot see inside Cython functions: their time is charged to the
Python caller. To make the Cython kernels visible, every function and class
exported by an ``allel2.opt.*`` module is replaced, wherever allel2 imported
it, by a thin Python wrapper defined in this file. The own time (``tottime``)
of a wrapper is then the time spent in the Cython kernel it wraps.
"""
import cProfile
import functools
import json
import pstats
import sys
import time
from collections import defaultdict

WRAPPER_FILE = __file__


def _rename(f, name):
    # cProfile identifies functions by their code object's name
    f.__code__ = f.__code__.replace(co_name=name, co_qualname=name)
    f.__name__ = f.__qualname__ = name


def _wrap_function(name, f):
    @functools.wraps(f)
    def cython_kernel(*args, **kwargs):
        return f(*args, **kwargs)
    _rename(cython_kernel, 'cython:' + name)
    return cython_kernel


def _wrap_iterator_class(name, cls):
    # e.g. VCFChunkIterator: time spent in __next__ is Cython parsing
    class CythonIterator:
        def __init__(self, *args, **kwargs):
            self._it = cls(*args, **kwargs)

        def __iter__(self):
            return self

        def __next__(self):
            # call the method directly so that cProfile reports it by name
            return self._it.__next__()
    _rename(CythonIterator.__next__, f'cython:{name}.__next__')
    return CythonIterator


def install_cython_timers():
    """Replace allel2.opt.* callables in every loaded allel2 module."""
    import allel2  # noqa: F401  (load the package and its Cython modules)
    replaced = {}
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith('allel2') or mod is None:
            continue
        for attr, obj in list(vars(mod).items()):
            origin = getattr(obj, '__module__', None) or ''
            if not origin.startswith('allel2.opt') or not callable(obj):
                continue
            if modname.startswith('allel2.opt'):
                continue
            key = (origin, attr)
            if key not in replaced:
                if isinstance(obj, type) and hasattr(obj, '__next__'):
                    replaced[key] = _wrap_iterator_class(attr, obj)
                elif isinstance(obj, type):
                    continue
                else:
                    replaced[key] = _wrap_function(attr, obj)
            setattr(mod, attr, replaced[key])
    return sorted(f'{origin}.{attr}' for origin, attr in replaced)


def _kernel_name(filename, funcname):
    """Name of the Cython kernel a profile entry belongs to, or None."""
    if filename == WRAPPER_FILE and funcname.startswith('cython:'):
        # own time of a wrapper is time spent inside the wrapped kernel
        return funcname[len('cython:'):]
    if filename.endswith('.pyx'):
        # modules compiled with profiling hooks report their functions directly
        return funcname
    if 'allel2.opt' in funcname:
        # methods of Cython classes, e.g. VCFChunkIterator.__next__
        return funcname.strip('<>')
    return None


def _category(filename, funcname):
    if _kernel_name(filename, funcname) is not None:
        return 'cython'
    f = filename.replace('\\', '/')
    if '/numpy/' in f or 'numpy' in funcname:
        return 'numpy'
    if '/zarr/' in f or '/numcodecs/' in f or 'zarr' in funcname or 'numcodecs' in funcname:
        return 'zarr/numcodecs'
    if '/dask/' in f:
        return 'dask'
    if '/scipy/' in f or '/sklearn/' in f:
        return 'scipy/sklearn'
    if '/pandas/' in f:
        return 'pandas'
    if '/allel2/' in f:
        return 'allel2 (python)'
    if filename == '~' or funcname.startswith('<'):
        # built-in functions and methods not attributable above
        return 'builtins'
    return 'other'


def profile_step(name, fn, *args, **kwargs):
    """Run ``fn`` under cProfile; return its result and a timing record."""
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        prof.disable()
    wall = time.perf_counter() - t0
    stats = pstats.Stats(prof).stats
    by_category = defaultdict(float)
    kernels = defaultdict(float)
    for (filename, _, funcname), (_, _, tottime, _, _) in stats.items():
        by_category[_category(filename, funcname)] += tottime
        kernel = _kernel_name(filename, funcname)
        if kernel is not None:
            kernels[kernel] += tottime
    profiled = sum(by_category.values())
    record = {
        'step': name,
        'wall_s': round(wall, 3),
        # profiler overhead inflates the profiled total; shares are relative to it
        'shares': {k: round(v / profiled, 4) for k, v in sorted(by_category.items())},
        'cython_kernels_s': {k: round(v, 3) for k, v in
                             sorted(kernels.items(), key=lambda kv: -kv[1])},
    }
    top = sorted(stats.items(), key=lambda kv: -kv[1][2])[:8]
    record['top_tottime'] = [
        {'func': f"{fn_.split('/')[-1]}:{func}", 'tottime_s': round(v[2], 3)}
        for (fn_, _, func), v in top]
    print(json.dumps({'step': name, 'wall_s': record['wall_s'],
                      'cython': record['shares'].get('cython', 0.0)}), flush=True)
    return result, record


def timed(name, fn, *args, **kwargs):
    """Wall time only, without the profiler (for unbiased totals)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    print(json.dumps({'step': name, 'wall_s': round(wall, 3), 'profiled': False}),
          flush=True)
    return result, {'step': name, 'wall_s': round(wall, 3)}
