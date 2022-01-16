from pathlib import Path
import numpy as np
import sqlite3


def main(dataset_name):

    conn = sqlite3.connect(f'{dataset_name}.db', check_same_thread=False)

    def new_delta(x, y):
        # return float(np.max(np.abs(x-y)))
        return float(np.mean(np.abs(x-y)))

    def get_inconsistency():
        _SQL = '''select rowid, model_id, backend_pair
                from inconsistency
                '''
        cur = conn.cursor()
        cur.execute(_SQL)
        return cur.fetchall()

    def get_localization_map(incons_id):
        _SQL = '''select rowid, layer_name, inbound_layers
                from localization_map
                where incons_id = ?
            '''
        cur = conn.cursor()
        cur.execute(_SQL, (incons_id,))
        return cur.fetchall()

    def update_inconsistency(incons_id, new_model_output_delta, new_loss_grads_delta):
        _SQL = '''update inconsistency
                set (model_output_delta, loss_grads_delta) = (?, ?)
                where rowid = ?
            '''
        conn.execute(_SQL, (new_model_output_delta, new_loss_grads_delta, incons_id,))
        conn.commit()

    def update_localization_map(_id, output_delta, output_R, grads_delta, grad_R):
        _SQL = '''update localization_map
                set (outputs_delta, outputs_R, gradients_delta, gradients_R) =(?, ?, ?, ?)
                where rowid = ?
            '''
        conn.execute(_SQL, (output_delta, output_R, grads_delta, grad_R, _id,))
        conn.commit()

    def get_last_layer_name(incons_id):
        _SQL = '''select layer_name, inbound_layers
                from localization_map
                where incons_id = ?
                '''
        cur = conn.cursor()
        cur.execute(_SQL, (incons_id,))
        a, b = set(), set()
        for layer_name, inbound_layers in cur.fetchall():
            a.add(layer_name)
            inbound_layers = set(eval(inbound_layers))
            b.update(inbound_layers)
        res = list(a - b)
        return res[0] if res else res

    def get_next_layer(incons_id, cur_layer):
        GET_NEXT = '''select layer_name
                    from localization_map
                    where incons_id == ? and inbound_layers like ?
                '''
        cur = conn.cursor()
        cur.execute(GET_NEXT, (incons_id, f'%{cur_layer}%',))
        res = cur.fetchone()
        return res[0] if res else res

    # update inconsistency table
    inconsistency = get_inconsistency()
    for incons_id, model_id, backend_pair in inconsistency:
        print(model_id)
        bk1, bk2 = backend_pair.split('_', 1)
        last_layer_name = get_last_layer_name(incons_id)
        exp_dir = Path(f'{dataset_name}_output') / str(model_id).zfill(6)

        bk1_output_dir = exp_dir / 'layer_outputs' / bk1
        bk2_output_dir = exp_dir / 'layer_outputs' / bk2
        bk1_grads_dir = exp_dir / 'layer_gradients' / bk1
        bk2_grads_dir = exp_dir / 'layer_gradients' / bk2

        if not last_layer_name:
            if bk1_grads_dir.exists():
                for fn in (exp_dir / 'loss_gradients' / bk1).glob("*.npy"):
                    last_layer_name = fn.stem
            else:
                continue
        if bk1_output_dir.exists() and bk2_output_dir.exists():
            # model_output_delta
            model_output_1 = np.load(str(exp_dir / 'layer_outputs' / bk1 / f'{last_layer_name}.npy'))
            model_output_2 = np.load(str(exp_dir / 'layer_outputs' / bk2 / f'{last_layer_name}.npy'))
            new_model_output_delta = new_delta(model_output_1, model_output_2)
        if bk1_grads_dir.exists() and bk2_grads_dir.exists():
            # loss_gradients_delta
            loss_grads_1 = np.load(str(exp_dir / 'loss_gradients' / bk1 / f'{last_layer_name}.npy'))
            loss_grads_2 = np.load(str(exp_dir / 'loss_gradients' / bk2 / f'{last_layer_name}.npy'))
            new_loss_grads_delta = new_delta(loss_grads_1, loss_grads_2)
            update_inconsistency(incons_id, new_model_output_delta, new_loss_grads_delta)

        # update localization_map
        localization_map = get_localization_map(incons_id)
        for _id, layer_name, inbound_layers in localization_map:
            output_delta = output_R = grads_delta = grad_R = None
            if bk1_output_dir.exists() and bk2_output_dir.exists():
                # 计算output_delta和R
                inbound_layers = eval(inbound_layers)
                output_1 = np.load(str(bk1_output_dir / f'{layer_name}.npy'))
                output_2 = np.load(str(bk2_output_dir / f'{layer_name}.npy'))
                output_delta = new_delta(output_1, output_2)

                if not inbound_layers or (len(inbound_layers) == 1 and inbound_layers[0][-6:] == '_input'):  # 无pre层
                    output_delta_pre = 0
                else:
                    delta_pre_list = []
                    for inbound_layer_name in inbound_layers:
                        # 计算每一pre层hidden_state的delta
                        pre_o1 = np.load(str(bk1_output_dir / f'{inbound_layer_name}.npy'))
                        pre_o2 = np.load(str(bk2_output_dir / f'{inbound_layer_name}.npy'))
                        delta_pre_list.append(new_delta(pre_o1, pre_o2))
                    output_delta_pre = max(delta_pre_list)

                output_R = (output_delta - output_delta_pre) / (output_delta_pre + 1e-7)

            if bk1_grads_dir.exists() and bk2_grads_dir.exists():
                # 计算gradients_delta和R
                next_layer = get_next_layer(incons_id, layer_name)
                grads_1 = np.load(str(bk1_grads_dir / f'{layer_name}.npy'))
                grads_2 = np.load(str(bk2_grads_dir / f'{layer_name}.npy'))
                grads_delta = new_delta(grads_1, grads_2)

                if not next_layer:  # 无next层，即最后一层的R是计算相对于loss_grads的变化率
                    grad_delta_next = new_loss_grads_delta
                else:
                    next_g1 = np.load(str(bk1_grads_dir / f'{next_layer}.npy'))
                    next_g2 = np.load(str(bk2_grads_dir / f'{next_layer}.npy'))
                    grad_delta_next = new_delta(next_g1, next_g2)

                grad_R = (grads_delta - grad_delta_next) / (grad_delta_next + 1e-7)
            update_localization_map(_id, output_delta, output_R, grads_delta, grad_R)


datasets = ['cifar10', 'mnist', 'fashion_mnist', 'imagenet', 'sinewave', 'price']

for dataset in datasets:
    main(dataset)
