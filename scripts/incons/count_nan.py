import sqlite3
import math
from pathlib import Path

threshold = 0.15
epsilon = 1e-5


def main(db_path):
    print(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)

    def get_incons_list():
        GET_INCONS = '''select rowid, *
                        from inconsistency
                    '''
        cur = conn.cursor()
        cur.execute(GET_INCONS)
        return cur.fetchall()

    def get_exec_status(model_id):
        GET_STATUS = '''select status
                        from model
                        where rowid = ?
                    '''
        cur = conn.cursor()
        cur.execute(GET_STATUS, (model_id,))
        res = cur.fetchone()[0]
        return eval(res)

    def get_localization_map(incons_id):
        GET_MAP = '''select *
                    from localization_map
                    where incons_id = ?
                '''
        cur = conn.cursor()
        cur.execute(GET_MAP, (incons_id,))
        return cur.fetchall()

    def get_loss_func(model_id):
        GET_LOSS = '''select loss_func
                    from model
                    where rowid == ?
                '''
        cur = conn.cursor()
        cur.execute(GET_LOSS, (model_id,))
        res = cur.fetchone()[0]
        return res

    incons_list = get_incons_list()

    def get_nan_backends(model_id):
        GET_NAN = '''select nan_backends
                     from model
                     where rowid == ?
        '''
        cur = conn.cursor()
        cur.execute(GET_NAN, (model_id,))
        res = cur.fetchone()[0]
        return eval(res) if res is not None else []

    def is_valid(s, phrase):
        if s == 0:
            return True
        if phrase == 'layers_output':
            return s >= 2
        elif phrase == 'loss':
            return s >= 3
        elif phrase == 'loss_grads':
            return s >= 4
        elif phrase == 'layers_grads':
            return s >= 5
        else:
            raise ValueError('Unknown phrase.')

    def simplify_layer_name(layer_name: str):
        tmp = layer_name.split('_')
        if tmp[-1] in ['normal', 'reduction']:
            return '_'.join(tmp[1:-1])
        else:
            return '_'.join(tmp[1:])

    nan_layer = {bk: set() for bk in ['tensorflow', 'theano', 'cntk']}

    for incons_id, model_id, bk_pair, model_output_delta, loss_delta, loss_grads_delta, _ in incons_list:
        status = get_exec_status(model_id)
        bk1, bk2 = tuple(bk_pair.split('_'))
        s1, s2 = int(status[bk1]), int(status[bk2])

        nan_backends = get_nan_backends(model_id)

        localization_map = get_localization_map(incons_id)
        if (is_valid(s1, 'layers_output') and is_valid(s2, 'layers_output')) and model_output_delta is None:
            for _, layer_name, output_delta, _, _, _, _, _ in localization_map:
                if output_delta is None:
                    for bk in nan_backends:
                        nan_layer[bk].add(simplify_layer_name(layer_name))
                    break

    nan_dir = Path(db_path).parent / 'layer_nan'
    nan_dir.mkdir(parents=True, exist_ok=True)
    for bk in ['tensorflow', 'theano', 'cntk']:
        with open(str(nan_dir / f"nan_{bk}.txt"), "w") as f:
            print(len(nan_layer[bk]), nan_layer[bk], file=f)
    return nan_layer


for i in range(1, 6):
    exp = 'E' + str(i)
    total_nan_layer = {bk: set() for bk in ['tensorflow', 'theano', 'cntk']}
    for fn in Path(exp).rglob("*.db"):
        nan_layer = main(str(fn))
        for bk in ['tensorflow', 'theano', 'cntk']:
            total_nan_layer[bk].update(nan_layer[bk])
    for bk in ['tensorflow', 'theano', 'cntk']:
        with open(str(Path(exp) / f"nan_{bk}_total.txt"), "w") as f:
            print(len(total_nan_layer[bk]), total_nan_layer[bk], file=f)
