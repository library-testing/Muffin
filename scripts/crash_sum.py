from pathlib import Path

backends = ['tensorflow', 'theano', 'cntk']
types = ['generation', 'detection']

crashes = {bk: {t: set() for t in types} for bk in backends}

for fn in Path(".").rglob("*.txt"):
    tmp = fn.stem.split('_')
    backend, t = tmp[0], tmp[1]
    with open(str(fn), "r") as f:
        for row in f.readlines()[1:]:
            msg = row[:-10].strip()
            if 'support' not in msg:
                crashes[backend][t].add(msg)

for bk in backends:
    for t in types:
        error_set = list(crashes[bk][t])
        with open(str(Path(".") / f"{bk}_{t}_total_error.txt"), "w") as f:
            print(len(error_set), file=f)
            error_set.sort()
            for e in error_set:
                print(e, file=f)
