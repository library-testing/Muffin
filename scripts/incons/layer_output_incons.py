from pathlib import Path
import sqlite3
import math


delta_threshold = 0.15
epsilon = 1e-5


def simplify_layer_name(layer_name: str):
    tmp = layer_name.split('_')
    if tmp[-1] in ['normal', 'reduction']:
        return '_'.join(tmp[1:-1])
    else:
        return '_'.join(tmp[1:])


def get_output_incons(conn, bk_pair):
    SELECT_OUTPUT_INCONS = '''select layer_name, outputs_R
                              from inconsistency, localization_map
                              where inconsistency.rowid == localization_map.incons_id and backend_pair == ? and outputs_delta > ?
                                    and (outputs_delta - 1e-7 * outputs_R) / (outputs_R + 1) < ?
                           '''
    cur = conn.cursor()
    cur.execute(SELECT_OUTPUT_INCONS, (bk_pair, delta_threshold, epsilon,))
    return cur.fetchall()


def get_incons(_dir, bk_pair, dataset_name):
    db_path = _dir / f'{dataset_name}.db'
    incons_set = set()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    for layer_name, _ in get_output_incons(conn, bk_pair):
        incons_set.add(simplify_layer_name(layer_name))
    incons_dir = _dir / 'layer_output_incons'
    incons_dir.mkdir(parents=True, exist_ok=True)
    with open(str(incons_dir / f"{dataset_name}_{bk_pair}.txt"), "w") as f:
        print(len(incons_set), incons_set, file=f)
    return incons_set


bk_pairs = ['tensorflow_theano', 'tensorflow_cntk', 'theano_cntk']

for i in range(1, 6):
    exp = 'E' + str(i)
    total_set = {bk_pair: set() for bk_pair in bk_pairs}
    for _dir in Path(exp).glob("*"):
        if _dir.is_dir():
            dataset_name = _dir.stem
            print(dataset_name)
            for bk_pair in bk_pairs:
                total_set[bk_pair].update(get_incons(_dir, bk_pair, dataset_name))
    for bk_pair in bk_pairs:
        with open(str(Path(exp) / f"{bk_pair}_total_layer_output_incons.txt"), "w") as f:
            print(len(total_set[bk_pair]), total_set[bk_pair], file=f)
