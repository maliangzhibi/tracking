"""
pytorch hook
来自 <http://www.tensorinfinity.com/paper_198.html> 

"""
import torch
import torch.nn as nn

def hook_fn(grad):
    g = 2*grad
    print(g)
    return g

x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
w = torch.Tensor([1, 2, 3, 4]).requires_grad_()

z = x + y
# z.retain_grad()

z.register_hook(hook_fn)

o = w.matmul(z)
o.backward()
# o.retain_grad()

print("x.requires_grad: ", x.requires_grad)
print("y.requires_grad: ", y.requires_grad)
print("z.requires_grad: ", z.requires_grad)
print("w.requires_grad: ", w.requires_grad)
print("o.requires_grad: ", o.requires_grad)

# z和o不是直接指定的变量，不属于PyTorch中的叶子节点，这样的变量属于中间变量；
# 虽然reuires_grad的参数都是True，但是反向传播后，只有叶子节点的梯度保存下
# 来，中间节点的梯度并不会保存下来，而是直接删除。所以不将中间变量加上retain_grad()
# 方法，中间变量的梯度就是None。
# 如想在反向传播后保存中间节点的梯度，则需要特殊指定：将以上代码做如下改变：
# 将中间节点之后加上retain_grad()。


print("x.grad: ", x.grad)
print("y.grad: ", y.grad)
print("z.grad: ", z.grad)
print("w.grad: ", w.grad)
print("o.grad: ", o.grad)

# 增加retain_grad()方法的坏处：增加内存占用
# 替代方案:使用hook保存中间变量梯度
# 对于中间变量z,hook的使用方式为:z.register_hook(hook_fn).其中hook_fn为用户自
# 定义的一个函数,其签名为: hook_fn(grad)->Tensor None
# 输入为z的梯度,输出为Tensor或None(一般用于直接打印梯度).反向传播时,梯度传播到了
# z,再继续向前传播之前,将会先传入hook_fn,如果返回值是None,则梯度值不改变,继续向前
# 传播,如果返回值为Tensor类型,则该Tensor将取代z的原有的梯度,向前传播.

# 一个变量还可以绑定多个hook_fn.


# ========== Hook for Modules ==========
# 网络模块module没有显示的变量名可以访问,被封装在神经网络中间.我们通常只能获得网络
# 的整体输入和输出,对于夹在网络中间的模块,只能在forward函数的返回值中包含中间module
# 的输出,或者其他很麻烦的方法(将网络按module名称拆分再组合),让中间层提取的feature暴露
# 出来.
# PyTorch有两种hook(分别为:register_forward_hook和register_backward_hook)来获取
# 正/反向传播的中间层模块输入和输出的feature/gradient,大大降低了获取模型内部信息流的难度.

# register_forward_hook: 对于模块module,使用方式为: module.register_forward_hook(hook_fn),
# 其中hook_fn的签名为: hook_fn(module, input, output)->None. 与Tensor的hook不同的是,module
# 没有任何返回值,即不能使用前向的hook来改变输入或输出的任何值.但是借助该该hook,我们可以方便地使用
# 预训练模型提取特征,示例代码如下:
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.initialize()
    
    def initialize(self):
        with torch.no_grad():
            self.fc1.weight = torch.nn.Parameter(
                torch.Tensor([[1., 2., 3.],
                [-4., -5., -6.],
                [7., 8., 9.],
                [-10., -11., -12.],])
            )
            self.fc1.bias = torch.nn.Parameter(torch.Tensor([1.0, 2.0, 3.0, 4.0]))
            self.fc2.weight = torch.nn.Parameter(torch.Tensor([[1.0, 2.0, 3.0, 4.0]]))
            self.fc2.bias = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        o = self.fc1(x)
        o = self.relu1(o)
        o = self.fc2(o)
        return o

# 全局变量,存储中间feature
total_feat_out = []
total_feat_in = []

def hook_fn_forward(module, input, output):
    '''
    定义forward hook function
    '''
    print(module)
    print('input', input)
    print('output', output)
    
    total_feat_out.append(output)
    total_feat_in.append(input)

total_grad_in = []
total_grad_out = []
def hook_fn_backward(moudle, grad_input, grad_output):
    '''
    定义backward hook function
    '''
    print(module)
    print('grad_output: ', grad_output)
    print('grad_input: ', grad_input)
    total_grad_in.append(grad_input)
    total_grad_out.append(grad_output)

model = Model()
model.register_backward_hook(hook_fn_backward)

modules = model.named_children()
for name, module in modules:
    print('name: ', name)
    module.register_forward_hook(hook_fn_forward)
    module.register_backward_hook(hook_fn_backward)

x = torch.Tensor([[1.0, 1.0, 1.0]]).requires_grad_()
o = model(x)
o.backward()
print('==========saved inputs and outputs==========')
for idx in range(len(total_feat_in)):
    print('input: ', total_feat_in[idx])
    print('output: ', total_feat_out[idx])
for idx in range(len(total_grad_in)):
    print('grad_output: ', total_grad_out[idx])
    print('grad_input: ', total_grad_in[idx])

x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
w = torch.Tensor([1, 2, 3, 4]).requires_grad_()

z = x + y
# z.retain_grad()

z.register_hook(lambda x: 2*x)
z.register_hook(lambda x: print(x))

o = w.matmul(z)
print('==========start backprop ==========')
o.backward()
print('==========end backprop ==========')

print("x.grad: ", x.grad)
print("y.grad: ", y.grad)
print("w.grad: ", w.grad)
print("z.grad: ", z.grad)

# 1. register_backward_hook与register_forward_hook类似,其作用是获取
# 神经网络反向传播过程中各个模块输入端和输出端地梯度值.对于module,使用
# 方式为: module.register_backward_hook(hook_fn), hook_fn签名为:
# hook_fn(module, grad_input, grad_output)->Tensor or None
# 输入端的梯度和输出端的梯度是站在前向传播的角度的.如:o = W * x+ b,其
# 中w/x/b都是输入端,o为输出端.输入或输出端有多个时,可以是tuple,此处则为
# 输入端包含w, x, b三个部分,输入端frad_input就包含三个元素的tuple.
# 2. 与forward hook的不同:
# 在forward hook中,input是x,而不包括w和b;
# 在backward hook中返回Tensor或None,而在forward hook中不返回.backward hook中
# 不能直接改变输入变量,但是可以返回新的grad_input,反向传播到它的上一个模块.

# 注意事项：
# register_backward_hook只能操作简单模块，不能操作包含多个子模块的复杂模块。对复
# 杂模块使用register_backward_hook，只能得到该模块最后一次简单操作的梯度信息。
# 对上面的代码修改，不再遍历各个子模块，而是把model整体绑在一个hook_fn_backward上：
# model = Model()
# model.register_backward_hook(hook_fn_backward)

# backward hook在全连接层和卷积层中表现不一致的地方：
# 1.形状：卷积层中，weight的梯度和weight的形状相同；在全连接层中，weight的梯度形状是weight形状的转置。
# 2.grad_input_tuple中各梯度的顺序：全连接层中，grad_input =  (对bias的导数， 对feature的导数，对w的导数)；
# 卷积层中：grad_input = (对feature的导数，对权重w的导数，对bias的导数)
# 3.batch_size>1时，对bias的梯度处理不同：卷积层中，对bias的梯度为整个batch的数据在bias上的梯度之和，grad_input=
# (对feature的导数，对权重w的导数，对bias的导数)；对于全连接层，对bias的梯度是分开的，batch中的每条数据，
# 对应于一个bias的梯度：grad_input = ((data1对bias的导数，data2对bias的导数...)，对feature的导数，对权重w的导数)。

# Guided Backpropsgstion：通过反向传播，计算需要可视化的输出或feature map对网络输入的梯度，归一化该梯度，作为图片
# 显示出来。梯度大的部分反映了输入图片该部分对目标输出的影响较大，反之较小。由此可以了解到神经网络所作的片段，到底受
# 图片哪些区域的影响，或者目标feature提取的是输入图片中哪些区域的特征。

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        self.model.eval()
        self.register_hooks()
    
    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            # 在全局变量中保存输入图片的梯度，该梯度由第一层卷积层反向传播得到，因此该函数
            # 需要绑定一个Conv2 Layer
            self.image_reconstruction = grad_in[0]
        
        def forward_hook_fn(module, input, output):
            # 在全局变量中保存ReLU层的前向输出，用于将来guided backpropagation
            self.activation_maps.append(output)
        
        def backward_hook_fn(module, grad_in, grad_out):
            # ReLU层的反向传播时，用其正向传播的输出作为guid
            # 反向传播和正向传播相反，先从后面传其
            grad = self.activation_maps.pop()
            grad[grad > 0] = 1

            # grad_in[0]表示feature的梯度，只保留大于0部分
            positive_grad_in = torch.clamp(grad_in[0], min=0.0)
            new_grad_in = positive_grad_in * grad
            # ReLU不含parameter，输入端是一个只有一个元素的tuple
            return (new_grad_in, )

        modules = list(self.model.features.named_children())
        print('==========modules==========：', modules)
        for name, module in modules:
            print('name: {}, module: {}'.format(name, module))
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
        # 对第一层卷积层注册hook
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)
        
    def visualize(self, input_image, target_class):
        # 获取输出
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        print('model_out: {}, model_out.shape: {}, model_out[pred_class]: {}'.format(model_output, 
        model_output.shape, model_output[0][pred_class]))
        print('pred_class:', pred_class)
         # 生成目标的one-hot向量，作为反向传播的起点
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)
        print('grad_target_map:', grad_target_map)
        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1
        model_output.backward(grad_target_map)
        result = self.image_reconstruction.data[0].permute(1, 2, 0)
        return result.numpy()

def normalize(I):
    # 归一化梯度map，mean=0,std=1
    norm = (I - I.mean())/I.std()
    # 将std重置为0.1,让梯度Map的数值尽可能接近0；均值加0.5，保证大部分梯度值为正;并将0,1以外的梯度值分别设置为0，1
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm

if __name__ == '__main__':
    from torchvision import models, transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = '../dataset/cat.jpg'
    I = Image.open(img_path).convert('RGB')
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = 224

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(I).unsqueeze(0).requires_grad_()
    model = models.alexnet(pretrained=True)
    guided_bp = Guided_backprop(model)
    result = guided_bp.visualize(tensor, None)

    result = normalize(result)
    plt.subplot(121)
    plt.imshow(I)
    plt.subplot(122)
    plt.imshow(result)
    plt.show()

    print('END')
