from pathlib import Path


a = 'BASELINE'
b = 'RMG'
c = 'LEMON'

a_layers = set()
b_layers = set()
c_layers = set()


def get_layers(name, layers):
    res = {bk: set() for bk in backends}
    for fn in Path(name).rglob("*.txt"):
        with fn.open(mode="r") as f:
            content = f.read()
        layers = eval(content)
        res.update(dict(layers))
    return res


# def get_layers(name, layers):
#     res = set()
#     for fn in Path(name).rglob("*.txt"):
#         with fn.open(mode="r") as f:
#             content = f.read()
#         start = content.find('{')
#         if start == -1:
#             continue
#         end = content.index('}', start)
#         layers = eval(content[start:end+1])
#         res.update(set(layers))
#     return res


backends = ['tensorflow', 'theano', 'cntk']

a_layers = get_layers(a, a_layers)
b_layers = get_layers(b, b_layers)
c_layers = get_layers(c, c_layers)

res = {bk: set() for bk in backends}

for bk in backends:
    res[bk] = a_layers[bk] - b_layers[bk] - c_layers[bk]

print(res)
