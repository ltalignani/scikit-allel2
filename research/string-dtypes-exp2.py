import numpy as np, pandas as pd, timeit, warnings
import allel2
T = np.dtypes.StringDType()
chrom = np.array(["2L"]*3+["2R"]*3, dtype=T); pos = np.array([1,5,9,1,5,9])
for name, c in [("StringDType", chrom), ("object", chrom.astype(object)), ("S", chrom.astype("S2"))]:
    try:
        idx = allel2.ChromPosIndex(c, pos)
        r = idx.locate_range("2R" if name!="S" else b"2R", 2, 9)
        print(name, "ChromPosIndex OK", r, "| unique:", np.unique(c), "| == :", (c == c[0]).sum())
    except Exception as e:
        print(name, "ERROR", type(e).__name__, str(e)[:120])
# dataframe
print("pd str storage:", pd.options.future.infer_string if hasattr(pd.options,'future') else None, pd.StringDtype().storage)
try:
    import pyarrow; print("pyarrow", pyarrow.__version__)
except ImportError: print("pyarrow absent")
df = pd.DataFrame({"CHROM": np.array(["2L","2R"], dtype=object), "T": np.array(["a","b"], dtype=T)})
print(df.dtypes.to_dict())
# perf on 1e6
n=10**6; base = np.array(["2L","2R","3L","3R","X"])[np.random.randint(0,5,n)]
for dt in [object, T, "S2", "U2"]:
    a = base.astype(dt)
    t = min(timeit.repeat(lambda: np.unique(a), number=1, repeat=3))
    t2 = min(timeit.repeat(lambda: (a == a[0]).sum(), number=1, repeat=3))
    print(f"{str(dt):14s} nbytes={a.nbytes/1e6:6.1f}MB unique={t*1e3:6.1f}ms eq={t2*1e3:6.1f}ms")
