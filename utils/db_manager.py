from typing import List
import sqlite3


class DbManager(object):

    def __init__(self, db_path: str):
        super().__init__()
        self.__conn = sqlite3.connect(db_path, check_same_thread=False)

    def register_model(self, dataset_name: str, node_num: int):
        '''将模型登记在model表中，并返回分配的model_id
        '''
        INSERT_A_MODEL = '''insert into model(dataset_name, node_num)
                            values(?, ?)
                         '''
        self.__conn.execute(INSERT_A_MODEL, (dataset_name, node_num,))
        self.__conn.commit()

        FETCH_ROWID = '''select last_insert_rowid() from model'''
        cur = self.__conn.cursor()
        cur.execute(FETCH_ROWID)
        return cur.fetchone()[0]

    def update_model_generate_fail_backends(self, model_id: int, fail_backends: List[str]):
        '''记录模型的generate_fail_backends
        '''
        UPDATE_GENERATE_FAIL_BKS = '''update model
                                      set generate_fail_backends = ?
                                      where rowid = ?
                                   '''
        self.__conn.execute(UPDATE_GENERATE_FAIL_BKS, (str(fail_backends), model_id,))
        self.__conn.commit()

    def update_model_crash_backends(self, model_id: int, crash_backends: List[str]):
        '''记录模型的crash_backends
        '''
        UPDATE_CRASH_BKS = '''update model
                              set crash_backends = ?
                              where rowid = ?
                           '''
        self.__conn.execute(UPDATE_CRASH_BKS, (str(crash_backends), model_id,))
        self.__conn.commit()

    def update_model_nan_backends(self, model_id: int, nan_backends: List[str]):
        '''记录模型的nan_backends
        '''
        UPDATE_NAN_BKS = '''update model
                            set nan_backends = ?
                            where rowid = ?
                         '''
        self.__conn.execute(UPDATE_NAN_BKS, (str(nan_backends), model_id,))
        self.__conn.commit()

    def update_model_inf_backends(self, model_id: int, inf_backends: List[str]):
        '''记录模型的inf_backends
        '''
        UPDATE_INF_BKS = '''update model
                            set inf_backends = ?
                            where rowid = ?
                          '''
        self.__conn.execute(UPDATE_INF_BKS, (str(inf_backends), model_id,))
        self.__conn.commit()

    def add_inconsistencies(self, incons: List[tuple]):
        INSERT_INCONS = '''insert into inconsistency(model_id, input_index, backend_pair, output_distance)
                           values(?, ?, ?, ?)
                        '''
        self.__conn.executemany(INSERT_INCONS, incons)
        self.__conn.commit()

    def get_incons_inputs_by_model_id_and_bk(self, model_id: int, backend: str, threshlod: float):
        SELECT_INCONS_INPUTS_AND_BKS_BY_MODEL_ID = '''select distinct input_index
                                                      from inconsistency
                                                      where model_id = ? and backend_pair like ? and (output_distance > ? or output_distance is null)
                                                      order by input_index asc
                                                   '''
        cur = self.__conn.cursor()
        cur.execute(SELECT_INCONS_INPUTS_AND_BKS_BY_MODEL_ID, (model_id, '%'+backend+'%', threshlod,))
        return [res[0] for res in cur.fetchall()]

    def get_incons_bk_pairs_by_model_id_and_inputs(self, model_id: int, input_index: int, threshold: float):
        SELECT_BK_PAIRS_BY_MODEL_ID_AND_INPUTS = '''select rowid, backend_pair
                                                    from inconsistency
                                                    where model_id = ? and input_index = ? and (output_distance > ? or output_distance is null)
                                                 '''
        cur = self.__conn.cursor()
        cur.execute(SELECT_BK_PAIRS_BY_MODEL_ID_AND_INPUTS, (model_id, input_index, threshold,))
        return cur.fetchall()

    def get_huge_incons(self, threshold: float, model_id: int):
        GET_HUGE_INCONS = '''select rowid, input_index, backend_pair
                             from inconsistency
                             where model_id = ? and (output_distance > ? or output_distance is null)
                             group by backend_pair
                          '''
        cur = self.__conn.cursor()
        cur.execute(GET_HUGE_INCONS, (model_id, threshold,))
        return cur.fetchall()

    def get_localization_map(self, incons_id):
        GET_LOCALIZATION_MAP = '''select *
                                  from localization_map
                                  where inconsistency_id == ?
                               '''
        cur = self.__conn.cursor()
        cur.execute(GET_LOCALIZATION_MAP, (incons_id,))
        return cur.fetchall()

    def add_training_incons(self, model_id, backend_pair, model_output_delta, loss_delta, loss_grads_delta):
        INSERT_INCONS = '''insert into inconsistency(model_id, backend_pair, model_output_delta, loss_delta, loss_grads_delta)
                           values(?, ?, ?, ?, ?)
                        '''
        self.__conn.execute(INSERT_INCONS, (model_id, backend_pair, model_output_delta, loss_delta, loss_grads_delta,))
        self.__conn.commit()

        FETCH_ROWID = '''select last_insert_rowid() from inconsistency'''
        cur = self.__conn.cursor()
        cur.execute(FETCH_ROWID)
        return cur.fetchone()[0]

    def record_loss_optimizer(self, model_id: int, loss: str, optimizer: str):
        '''记录模型的train配置
        '''
        UPDATE_LOSS_OPTIMIZER = '''update model
                                   set loss_func = ?, optimizer = ?
                                   where rowid = ?
                                '''
        self.__conn.execute(UPDATE_LOSS_OPTIMIZER, (loss, optimizer, model_id,))
        self.__conn.commit()

    def record_status(self, model_id: int, status: list):
        UPDATE_STATUS = '''update model
                           set status = ?
                           where rowid = ?
                        '''
        self.__conn.execute(UPDATE_STATUS, (str(status), model_id,))
        self.__conn.commit()

    def get_model_info(self, model_id: int):
        GET_MODEL_INFO = '''select *
                            from model
                            where rowid = ?
                         '''
        cur = self.__conn.cursor()
        cur.execute(GET_MODEL_INFO, (model_id,))
        return cur.fetchone()

    def update_losses(self, model_id: int, bk_pair: str, loss_delta: float):
        UPDATE_LOSS = '''update inconsistency
                         set loss_delta = ?
                         where model_id = ? and backend_pair = ?
                      '''
        self.__conn.execute(UPDATE_LOSS, (loss_delta, model_id, bk_pair,))
        self.__conn.commit()

    def add_localization_map(self, infos):
        INSERT_LOCALIZATION_MAP = '''insert into localization_map(incons_id, layer_name, outputs_delta, outputs_R, gradients_delta, gradients_R, inbound_layers)
                               values(?, ?, ?, ?, ?, ?, ?)
                            '''
        self.__conn.executemany(INSERT_LOCALIZATION_MAP, infos)
        self.__conn.commit()
