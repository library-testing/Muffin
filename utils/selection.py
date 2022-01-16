from typing import List
import numpy as np
import random


class Roulette(object):

    class Element(object):
        def __init__(self, name: str, selected: int = 0):
            self.name = name
            self.selected = selected

        def record(self):
            self.selected += 1

        @property
        def score(self):
            return 1.0 / (self.selected + 1)

    def __init__(self, layer_types: List[str], layer_conditions: dict, use_heuristic: bool = True):
        self.__pool = {name: self.Element(name=name) for name in layer_types}
        self.__layer_conditions = layer_conditions
        self.__use_heuristic = use_heuristic

    def update(self, name):
        if name == 'input_object':
            return
        for n, el in self.__pool.items():
            if n == name:
                el.record()
                return

    def coverage(self):
        selected_map = {name: el.selected for name, el in self.__pool.items()}
        cnt = 0
        for selected in selected_map.values():
            if selected > 0:
                cnt += 1
        coverage_rate = cnt / len(selected_map)
        return coverage_rate, selected_map

    def choose_element(self, pool: List[str], **kwargs):
        candidates = []
        _sum = 0
        for el_name in pool:
            cond = self.__layer_conditions.get(el_name, None)
            if cond is None or cond(**kwargs):  # available的layer
                candidates.append(self.__pool[el_name])
                _sum += self.__pool[el_name].score

        if self.__use_heuristic:
            rand_num = np.random.rand() * _sum
            for el in candidates:
                if rand_num < el.score:
                    return el.name
                else:
                    rand_num -= el.score
        else:
            return random.choice(candidates).name


# class MCMC(object):

#     class Element(object):
#         def __init__(self, name: str, select_model_cnt: int = 0):
#             self.name = name
#             self.__select_model_cnt = select_model_cnt

#         def update(self):
#             self.__select_model_cnt += 1

#         @property
#         def score(self):
#             return 1 / (self.__select_model_cnt + 1)

#     def __init__(self, layer_types: List[str], layer_conditions: dict):
#         self.__p = 1 / len(layer_types)
#         self.__normal_pool = [self.Element(name=name) for name in layer_types]
#         self.__layer_conditions = layer_conditions

#     @property
#     def elements(self):
#         return {el.name: el for el in self.__normal_pool}

#     def choose_element(self, e1: Optional[str], **kwargs):
#         if e1 is None:  # 第一次选择
#             return self.__normal_pool[np.random.randint(0, len(self.__normal_pool))].name
#         else:
#             self.sort_elements()
#             k1 = self.index(e1)
#             k2 = -1
#             prob = 0
#             while np.random.rand() >= prob:  # prob为接受率
#                 k2 = np.random.randint(0, len(self.__normal_pool))
#                 cond = self.__layer_conditions.get(self.__normal_pool[k2].name, None)
#                 if cond is not None and not cond(e1=e1, **kwargs):  # 拒绝接不上的layer
#                     continue
#                 prob = (1 - self.__p) ** (k2 - k1)
#             return self.__normal_pool[k2].name

#     def sort_elements(self):
#         import random
#         random.shuffle(self.__normal_pool)
#         self.__normal_pool.sort(key=lambda el: el.score, reverse=True)

#     def index(self, el_name):
#         for i, el in enumerate(self.__normal_pool):
#             if el.name == el_name:
#                 return i
#         return -1
