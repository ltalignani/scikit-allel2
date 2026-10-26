import numpy as np, zarr, pandas as pd, json, os, shutil, warnings
print("numpy", np.__version__, "zarr", zarr.__version__, "pandas", pd.__version__)
obj = np.array(["20", "X", "HLA-A*01"], dtype=object)
S = np.array([b"20", b"X", b"HLA"], dtype="S10")
U = np.array(["20", "X", "ab"], dtype="U8")
T = np.array(["20", "X", "ab"], dtype=np.dtypes.StringDType())
for fmt in (3, 2):
    for name, a, dt in [("obj->'str'", obj, "str"), ("obj->object", obj, object), ("S10", S, S.dtype), ("U8", U, U.dtype), ("StringDType", T, T.dtype)]:
        p = f"t{fmt}.zarr"; shutil.rmtree(p, ignore_errors=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                z = zarr.create_array(p, shape=a.shape, dtype=dt, zarr_format=fmt)
                z[:] = a
                r = zarr.open_array(p)[:]
                meta = json.load(open(f"{p}/zarr.json" if fmt == 3 else f"{p}/.zarray"))
                d = meta.get("data_type", meta.get("dtype"))
                print(f"v{fmt} {name:14s} meta={d!s:40.40s} read={r.dtype!r} {r[:2]} warn={[str(x.message)[:60] for x in w][:1]}")
            except Exception as e:
                print(f"v{fmt} {name:14s} ERROR {type(e).__name__}: {str(e)[:100]}")
s = pd.Series(obj); print("pandas Series(obj).dtype", s.dtype, "| from S:", pd.Series(S).dtype, "| from T:", pd.Series(T).dtype)
print("np.array(list of str).dtype", np.array(["a"]).dtype)
