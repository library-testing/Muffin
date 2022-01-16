from typing import Optional


class ModelTemplate(object):

    class Cell(object):
        def __init__(self, topo_id: int, _type: str, node_num: int, is_input: bool = False, is_output: bool = False,
                     input_shapes: Optional[tuple] = None, output_shapes: Optional[tuple] = None):
            self.id = topo_id
            self.type = _type
            self.node_num = node_num
            self.is_input = is_input
            self.is_output = is_output
            self.input_shapes = input_shapes
            self.output_shapes = output_shapes

            self.inbound_cells = []
            self.outbound_cells = []

        def connect_to(self, cells):
            for c2 in cells:
                c2.inbound_nodes.append(self)
                self.outbound_nodes.append(c2)

    def __init__(self, randomer, template_type, cell_num, node_num_per_normal_cell, node_num_per_reduction_cell, input_shape, output_shape, **kwargs):
        self.__random = randomer

        self.cells = self.__init_T1(cell_num, node_num_per_normal_cell, node_num_per_reduction_cell, input_shape, output_shape)

    def __init_T1(self, cell_num, node_num_per_normal_cell, node_num_per_reduction_cell, input_shape, output_shape):
        dim = self.__random.randint_in_range((3, 5))
        if input_shape is None:
            input_shape = self.__random.shape()
        if output_shape is None:
            output_shape = self.__random.shape()

        self.input_shape = input_shape
        self.output_shape = output_shape

        cells = []
        cur_shapes = [input_shape]
        for i in range(0, cell_num):
            output_shapes = [self.__random.shape(dim=dim)]
            cell = self.Cell(i,
                             _type='normal' if i % 2 == 0 else 'reduction',
                             node_num=node_num_per_normal_cell if i % 2 == 0 else node_num_per_reduction_cell,
                             is_input=(i == 0),
                             is_output=(i == cell_num - 1),
                             input_shapes=cur_shapes,
                             output_shapes=output_shapes)
            cur_shapes = output_shapes
            cells.append(cell)
        return cells
