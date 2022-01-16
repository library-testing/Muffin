from typing import Optional
import random


class DAG(object):

    class Node(object):
        def __init__(self, topo_id: int, is_input: bool = False, is_output: bool = False, output_shape: Optional[tuple] = None):
            self.id = topo_id
            self.is_input = is_input
            self.is_output = is_output
            self.output_shape = output_shape
            self.type = None

            self.inbound_nodes = []
            self.outbound_nodes = []

        @property
        def is_merging(self):
            return len(self.inbound_nodes) > 1

        @property
        def is_connected(self):
            return bool(self.inbound_nodes)

        def remove(self):
            n1 = self.inbound_nodes[0]
            n1.connect_to(self.outbound_nodes)
            for n2 in self.outbound_nodes:
                # print(f"{self.id} -x-> {n2.id}")
                n2.inbound_nodes.remove(self)

        def connect_to(self, nodes):
            for n2 in nodes:
                # print(f"{self.id} --> {n2.id}")
                n2.inbound_nodes.append(self)
                self.outbound_nodes.append(n2)

    def __init__(self, main_node_num: int, input_shapes: list, output_shapes: list, max_branch_num: int):
        if len(input_shapes) + len(output_shapes) > main_node_num:
            raise ValueError("Dag's node num is not enough.")

        self.__max_branch_num = max_branch_num

        # 选取dag中的input结点和output结点
        sampled_id = random.sample(range(1, main_node_num-1), k=len(input_shapes)+len(output_shapes)-2)
        random.shuffle(sampled_id)
        inputs_id = sorted([0] + sampled_id[:len(input_shapes)-1])
        outputs_id = sorted(sampled_id[len(input_shapes)-1:] + [main_node_num-1])
        # print(f"inputs_id: {inputs_id}  outputs_id: {outputs_id}")
        self.nodes = []
        for i, output_shape in zip(inputs_id, input_shapes):
            self.nodes.append(self.Node(i, is_input=True, output_shape=output_shape))
        for i, output_shape in zip(outputs_id, output_shapes):
            self.nodes.append(self.Node(i, is_output=True, output_shape=output_shape))
        for i in range(main_node_num):
            if i not in inputs_id and i not in outputs_id:
                self.nodes.append(self.Node(i))
        self.nodes.sort(key=lambda n: n.id)  # 拓扑排序
        self.__generate()
        self.show()

    def __generate(self, adjoin_prob: float = 0.9):
        # 从每个非output出发向后连至少一个，连接对象必须非input
        for cur_id, n1 in enumerate(self.nodes):
            if not n1.is_output:
                if not self.nodes[cur_id+1].is_input and random.random() < adjoin_prob:  # 大概率直接连接下一个
                    targets = [self.nodes[cur_id+1]]
                else:
                    targets = [n2 for n2 in self.nodes[cur_id+1:] if not n2.is_input]
                n1.connect_to(random.sample(targets, random.randint(1, min(len(targets), self.__max_branch_num-1))))
        # 从每个非input出发向前连至少一个，连接对象必须非output (前一步已经被连的可以不连) [这一步只是补充作用，以保证所有output都有被连接]
        for cur_id, n2 in enumerate(self.nodes):
            if not n2.is_input and not n2.is_connected:
                if not self.nodes[cur_id-1].is_output and random.random() < adjoin_prob:  # 大概率直接连接上一个
                    targets = [self.nodes[cur_id-1]]
                else:
                    targets = [n1 for n1 in self.nodes[:cur_id] if not n1.is_output]
                random.choice(targets).connect_to([n2])

    def show(self):
        for n1 in self.nodes:
            print(f"n1: {n1.id}  n2: {[n2.id for n2 in n1.outbound_nodes]}")
