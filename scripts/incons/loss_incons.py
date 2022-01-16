from pathlib import Path
import sqlite3
import math


threshold = 0.15
epslion = 1e-5


def get_loss_incons(conn, bk_pair):
    SELECT_LOSS_INCONS = '''select model_id, loss_func, loss_delta
                            from model, inconsistency
                            where model.rowid == inconsistency.model_id and backend_pair == ? and model_output_delta < ? and loss_delta > ?
                         '''
    cur = conn.cursor()
    cur.execute(SELECT_LOSS_INCONS, (bk_pair, epslion, threshold))
    return cur.fetchall()


def get_incons(_dir, bk_pair, dataset_name):
    db_path = _dir / f'{dataset_name}.db'
    incons_set = set()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    for _, loss_func, _ in get_loss_incons(conn, bk_pair):
        incons_set.add(loss_func)
    incons_dir = _dir / 'loss_incons'
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
        with open(str(Path(exp) / f"{bk_pair}_total_loss_incons.txt"), "w") as f:
            print(len(total_set[bk_pair]), total_set[bk_pair], file=f)
