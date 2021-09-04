from pathlib import Path
import sqlite3


delta_threshold = 0.15
epsilon = 1e-5


def simplify_layer_name(layer_name: str):
    tmp = layer_name.split('_')
    if tmp[-1] in ['normal', 'reduction']:
        return '_'.join(tmp[1:-1])
    else:
        return '_'.join(tmp[1:])


def get_output_incons(conn, bk_pair, model_id):
    SELECT_OUTPUT_INCONS = '''select layer_name, outputs_R
                              from inconsistency, localization_map
                              where inconsistency.rowid == localization_map.incons_id
                                    and model_id = ? and backend_pair == ?
                                    and outputs_delta > ? and (outputs_delta - 1e-7 * outputs_R) / (outputs_R + 1) < ?
                           '''
    cur = conn.cursor()
    cur.execute(SELECT_OUTPUT_INCONS, (model_id, bk_pair, delta_threshold, epsilon,))
    return cur.fetchall()


def get_incons(conn, bk_pair, model_id):
    incons_set = set()
    for layer_name, _ in get_output_incons(conn, bk_pair, model_id):
        incons_set.add(simplify_layer_name(layer_name))
    return incons_set


def get_model_num(conn):
    _SQL = '''select rowid
              from model
            '''
    cur = conn.cursor()
    cur.execute(_SQL)
    return cur.fetchall()

bk_pairs = ['tensorflow_theano', 'tensorflow_cntk', 'theano_cntk']
backends = ['tensorflow', 'theano', 'cntk']

total_bug = {bk: set() for bk in backends}


for i in range(1, 6):
    exp = 'E' + str(i)
    for _dir in Path(exp).glob("*"):
        if not _dir.is_dir():
            continue
        dataset_name = _dir.name

        db_path = _dir / f'{dataset_name}.db'
        conn = sqlite3.connect(str(db_path), check_same_thread=False)

        model_ids = get_model_num(conn)

        for model_id in model_ids:
            model_id = model_id[0]
            pair_incons = {bk_pair: set() for bk_pair in bk_pairs}
            for bk_pair in bk_pairs:
                pair_incons[bk_pair] = get_incons(conn, bk_pair, model_id)

            total_incons = set()
            for incons_set in pair_incons.values():
                total_incons.update(incons_set)

            bk_bug = {bk: set() for bk in backends}
            for layer_name in total_incons:
                tf = 0
                th = 0
                ck = 0
                if layer_name in pair_incons['tensorflow_theano']:
                    tf += 1
                    th += 1
                if layer_name in pair_incons['tensorflow_cntk']:
                    tf += 1
                    ck += 1
                if layer_name in pair_incons['theano_cntk']:
                    th += 1
                    ck += 1
                if tf > th and tf > ck:
                    bk_bug['tensorflow'].add(layer_name)
                if th > tf and th > ck:
                    bk_bug['theano'].add(layer_name)
                if ck > tf and ck > th:
                    bk_bug['cntk'].add(layer_name)

            has_bug = False
            for bk in backends:
                if bk_bug[bk]:
                    has_bug = True
                    total_bug[bk].update(bk_bug[bk])
            # if has_bug:
            #     with open("FC_bugs.txt", "a") as f:
            #         print(f"{dataset_name} model_id={model_id}  {bk_bug}", file=f)

with open("FC_total_bugs.txt", "a") as f:
    print(total_bug, file=f)
