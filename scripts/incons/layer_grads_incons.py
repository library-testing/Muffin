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


def get_next_layer(conn, incons_id, cur_layer):
    GET_NEXT = '''select layer_name
                  from localization_map
                  where incons_id == ? and inbound_layers like ? and outputs_delta < ?
               '''
    cur = conn.cursor()
    cur.execute(GET_NEXT, (incons_id, f'%{cur_layer}%', epsilon,))
    res = cur.fetchone()
    return res[0] if res else res


def get_grads_incons(conn, bk_pair):
    SELECT_GRADS_INCONS = '''select inconsistency.rowid, layer_name, gradients_R
                              from inconsistency, localization_map
                              where inconsistency.rowid == localization_map.incons_id and backend_pair == ?
                              and gradients_delta > ?
                              and (gradients_delta - 1e-7 * gradients_R) / (gradients_R + 1) < ?
                              and loss_delta < ? and loss_grads_delta < ?
                          '''
    cur = conn.cursor()
    cur.execute(SELECT_GRADS_INCONS, (bk_pair, delta_threshold, epsilon, epsilon, epsilon,))
    return cur.fetchall()


def get_loss_grads_incons(conn, bk_pair):
    GET_LOSS_GRADS_INCONS = '''select loss_func, model.rowid
                               from model, inconsistency
                               where model.rowid == inconsistency.model_id and backend_pair == ?
                               and loss_grads_delta > ? and loss_delta < ?
                            '''
    cur = conn.cursor()
    cur.execute(GET_LOSS_GRADS_INCONS, (bk_pair, delta_threshold, epsilon,))
    return cur.fetchall()


def get_incons(_dir, bk_pair, dataset_name):
    db_path = _dir / f'{dataset_name}.db'
    incons_set = set()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    for incons_id, layer_name, _ in get_grads_incons(conn, bk_pair):
        bug_layer = get_next_layer(conn, incons_id, layer_name)
        if bug_layer:
            incons_set.add(simplify_layer_name(bug_layer))
    for loss_func, _ in get_loss_grads_incons(conn, bk_pair):
        incons_set.add(loss_func)
    incons_dir = _dir / 'layer_grads_incons'
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
        with open(str(Path(exp) / f"{bk_pair}_total_layer_grads_incons.txt"), "w") as f:
            print(len(total_set[bk_pair]), total_set[bk_pair], file=f)
