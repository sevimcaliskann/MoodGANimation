from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
from .networks import NetworksFactory
from .generator_wasserstein_gan import ResidualBlock
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot


# For traces

def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    """ Produces graphs of torch.jit.trace outputs
    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = make_dot_from_trace(trace)
    """
    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def print_grads(var):
    if hasattr(var, 'next_functions'):
        count = 0
        for u in var.next_functions:
            print('count: ', count)
            count = count+1
            if u[0] is not None:
                print(type(u[0]).__name__)
                print_grads(u[0])

def size_to_str(size):
    return '(' + (', ').join(['%d' % v for v in size]) + ')'

def name_nodes(layer):
    if isinstance(layer, torch.nn.Conv2d):
        u = layer._parameters['weight']
        node_name = '%s\n %s' % (str(type(layer).__name__), size_to_str(u.size()))
        #u = layer._parameters['weight'].requires_grad
        #node_name = '%s\n %s' % (str(type(layer).__name__), str(u))
    elif isinstance(layer, torch.nn.InstanceNorm2d):
        weight = layer._parameters['weight']
        bias = layer._parameters['bias']
        node_name = '%s\n Weight: %s \n Bias: %s ' % (str(type(layer).__name__), size_to_str(weight.size()), size_to_str(bias.size()))
        #weight = layer._parameters['weight'].requires_grad
        #bias = layer._parameters['bias'].requires_grad
        #node_name = '%s\n Weight: %s \n Bias: %s ' % (str(type(layer).__name__), str(weight), str(bias))
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        u = layer._parameters['weight']
        node_name = '%s\n %s' % (str(type(layer).__name__), size_to_str(u.size()))
        #u = layer._parameters['weight'].requires_grad
        #node_name = '%s\n %s' % (str(type(layer).__name__), str(u))
    else:
        node_name = str(type(layer).__name__)
    return node_name


def visualize_generator(node_attr, gen_net):
    last_node = None
    for seq in gen_net.children():
        if isinstance(seq, torch.nn.Sequential):
            prev = None
            res_track = None
            for layer in seq.children():
                if isinstance(layer, ResidualBlock):
                    res_prev = None
                    if res_track is not None:
                        first_node = list(list(layer.children())[0].children())[0]
                        dot.edge(str(id(res_track)), str(id(first_node)))
                    res_track = prev
                    for child in list(layer.children())[0].children():
                        node_name = name_nodes(child)
                        dot.node(str(id(child)), node_name)
                        if res_prev is not None:
                            dot.edge(str(id(res_prev)), str(id(child)))
                        elif res_prev is None and prev is not None:
                            dot.edge(str(id(prev)), str(id(child)))
                        res_prev = child
                        prev = child
                            #
                else:
                    node_name = name_nodes(layer)
                    dot.node(str(id(layer)), node_name)
                    if prev is not None:
                        dot.edge(str(id(prev)), str(id(layer)))
                    if res_track is not None:
                        dot.edge(str(id(res_track)), str(id(layer)))
                        res_track = None
                    prev = layer



    last_node = list(gen_net.main.children())[-1]
    img_reg = list(gen_net.img_reg.children())[0]
    att_reg = list(gen_net.attetion_reg.children())[0]
    dot.edge(str(id(last_node)), str(id(img_reg)))
    dot.edge(str(id(last_node)), str(id(att_reg)))
    dot.render('gen_graph', view=True)


def visualize_discriminator(node_attr, disc_net):
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()


    last_node = None
    for seq in disc_net.children():
        if isinstance(seq, torch.nn.Sequential):
            prev = None
            res_track = None
            for layer in seq.children():
                if isinstance(layer, ResidualBlock):
                    res_prev = None
                    if res_track is not None:
                        first_node = list(list(layer.children())[0].children())[0]
                        dot.edge(str(id(res_track)), str(id(first_node)))
                    res_track = prev
                    for child in list(layer.children())[0].children():
                        node_name = name_nodes(child)
                        dot.node(str(id(child)), node_name)
                        if res_prev is not None:
                            dot.edge(str(id(res_prev)), str(id(child)))
                        elif res_prev is None and prev is not None:
                            dot.edge(str(id(prev)), str(id(child)))
                        res_prev = child
                        prev = child
                            #
                else:
                    node_name = name_nodes(layer)
                    dot.node(str(id(layer)), node_name)
                    if prev is not None:
                        dot.edge(str(id(prev)), str(id(layer)))
                    if res_track is not None:
                        dot.edge(str(id(res_track)), str(id(layer)))
                        res_track = None
                    prev = layer
        else:
            node_name = name_nodes(seq)
            dot.node(str(id(seq)), node_name)



    last_node = list(disc_net.main.children())[-1]
    real = disc_net.conv1
    aux = disc_net.conv2
    dot.edge(str(id(last_node)), str(id(real)))
    dot.edge(str(id(last_node)), str(id(aux)))
    dot.render('disc_graph', view=True)



transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                           std=[0.5, 0.5, 0.5])
                                      ])


input = np.random.rand(3, 128, 128).astype(dtype=np.float32)
input = torch.unsqueeze(torch.from_numpy(input), 0)
gen_net = NetworksFactory.get_by_name('generator_wasserstein_gan', c_dim=17)
disc_net = NetworksFactory.get_by_name('discriminator_wasserstein_gan', c_dim=17)
c = np.zeros(17)
c[16] = 0.8
c = c.astype(dtype = np.float32)
c = torch.unsqueeze(torch.from_numpy(c), 0)

#c = c.unsqueeze(2).unsqueeze(3)
#c = c.expand(c.size(0), c.size(1), input.size(2), input.size(3))
#x = torch.cat([input, c], dim=1)
#features = gen_net.main(x)

#print('type of features: ', type(features))
#print('size of features: ', features.size())

#y, y_mask = gen_net(input, c)
#grad = y.mean().grad_fn
#dot = make_dot((y.mean(), y_mask.mean()), params=dict(gen_net.named_parameters()))
#dot.render('graph', view=True)
node_attr = dict(style='filled',
                 shape='box',
                 align='left',
                 fontsize='12',
                 ranksep='0.1',
                 height='0.2')
dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
seen = set()
visualize_discriminator(node_attr, disc_net)
visualize_generator(node_attr, gen_net)


for layer in gen_net.main:
    print(type(layer).__name__)
