from pathlib import Path
import sqlite3


threshold = 0.15
epslion = 1e-5


def get_loss_incons(conn, bk_pair, model_id):
    SELECT_LOSS_INCONS = '''select model_id, loss_func, loss_delta
                            from model, inconsistency
                            where model.rowid == inconsistency.model_id
                                  and model_id = ? and backend_pair == ?
                                  and model_output_delta < ? and loss_delta > ?
                         '''
    cur = conn.cursor()
    cur.execute(SELECT_LOSS_INCONS, (model_id, bk_pair, epslion, threshold))
    return cur.fetchall()


def get_incons(conn, bk_pair, model_id):
    incons_set = set()
    for _, loss_func, _ in get_loss_incons(conn, bk_pair, model_id):
        incons_set.add(loss_func)
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
            #     with open("LC_bugs.txt", "a") as f:
            #         print(f"{dataset_name} model_id={model_id}  {bk_bug}", file=f)

with open("LC_total_bugs.txt", "a") as f:
    print(total_bug, file=f)
