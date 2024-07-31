import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, Dropout, Softmax, Tanh, Sigmoid

#-------------------------对类进行继承和重新定义
class MLP(torch.nn.Module) :
    def __init__(self, input_n, output_n, num_layer, layer_list, dropout=0.1) :
        super(MLP, self).__init__()
        self.input_n    = input_n
        self.outout_n   = output_n
        self.num_layer  = num_layer
        self.layer_lsit = layer_list

        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=True),
            Sigmoid()
        )
        self.hidden_layer = Sequential()

        for index in range(num_layer-1) :
            self.hidden_layer.extend([Linear(layer_list[index], layer_list[index+1], bias=True), Sigmoid()])

        self.dropout = Dropout(dropout)

        self.output_layer = Sequential(
            Linear(layer_list[-1], output_n, bias=True)
            # ReLU()
            # Softmax(dim=1)
        )

    #-------------------此处forward函数强烈建议命名为forward，由于魔术方法的存在，会自动调用forward方法，
    #-------------------即object.forward(x) 的作用等于 object(x) 但不建议写object.forward(x)
    #-------------------因为由于魔术方法的存在，这样写会导致这个方法调用两次，详细机理见收藏的链接
    def forward(self, x) :                       
        in_put = self.input_layer(x)
        hidden = self.hidden_layer(in_put)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        output = output.view(-1)
        return output
