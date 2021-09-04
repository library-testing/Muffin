from pathlib import Path

backends = ['tensorflow', 'theano', 'cntk']
types = ['generation', 'detection']

crashes = {bk: {t: [] for t in types} for bk in backends}


def parse(segment):
    backend = None
    for bk in backends:
        if bk in segment[:50]:
            backend = bk
            break
    if backend is None:
        print(segment)
        raise ValueError("Backend Parse Fail.")
    keywords = ['ValueError: ', 'RuntimeError: ', 'TypeError: ', 'tensorflow.python.framework.errors_impl.InvalidArgumentError: ',
                'RecursionError: ', 'NotImplementedError: ', 'AttributeError: ', 'Exception: ']
    for kw in keywords:
        start = segment.find(kw)
        if start == -1:
            continue
        end = segment.find('\n', start)
        return backend, segment[start: end].strip()
    print(segment)
    raise ValueError("Error Msg Not Find.")


def error_filter(msg):
    kw = ['support', 'categorical_crossentropy']
    for k in kw:
        if k in msg:
            return False
    return True


for output_dir in Path(".").glob("*output"):
    for _dir in output_dir.glob("*"):
        if _dir.is_dir():
            log_dir = _dir / 'logs'
            if log_dir.exists():
                for fn in log_dir.glob("*log"):
                    if fn.stem in types:
                        with fn.open(mode="r", encoding='utf-8') as f:
                            content = f.read()
                        segments = content.split('[ERROR]')
                        for segment in segments:
                            if segment == '':
                                continue
                            backend, error_msg = parse(segment)
                            if error_filter(error_msg):
                                crashes[backend][fn.stem].append(error_msg + f"  exp_id={int(_dir.name)}")
    for bk in backends:
        for t in types:
            error_set = crashes[bk][t]
            _dir = Path(".") / 'crashes' / output_dir.name[:-7]
            _dir.mkdir(parents=True, exist_ok=True)
            with open(str(_dir / f"{bk}_{t}_error.txt"), "w") as f:
                print(len(error_set), file=f)
                error_set.sort()
                for e in error_set:
                    print(e, file=f)
