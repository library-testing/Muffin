import sys
import pickle

fn = sys.argv[1]

with open(fn, "rb") as f:
    a = pickle.load(f)
	
with open(fn[:-4]+".txt", "w") as f:
    f.write(str(list(a)))
