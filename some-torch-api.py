import torch
a = torch.randn(3, 3)
a1 = torch.randn(3, 3)
b = torch.triu(a, diagonal=-1)   # 上三角，参数diagonal是与对角线的偏移
c = torch.tril(a)   # 下三角
d = torch.diag(a) # 对角线组成的向量，参数diagonal是与对角线的偏移

e = torch.flip(a, dims=[0,1])   # 先0后1（先1后0是逆操作，可以还原）
f = torch.flip(e, dims=[1,0])

abs = torch.ones(3)  # 模长
angle = torch.arange(0, 3) * torch.pi / 2    # 幅角（x正轴逆时针旋转的弧度）
complex_planar = torch.polar(abs=abs, angle=angle)   # 将极坐标表示的复数，转换为直角坐标表示的复数
complex_planar_ = torch.view_as_complex(torch.stack([complex_planar.real, complex_planar.imag], dim=1))   # 提取实部与虚部，并用view_as_complex表示为复数
real_planar =  torch.view_as_real(complex_planar)   # 重新作为实数
# torch.conj(input): 用于计算输入张量的共轭复数。
# torch.angle(input): 返回输入张量的复数角度。
# torch.abs(input): 返回输入张量的绝对值，如果输入是复数，则返回复数的模。
# torch.polar(input): 将输入张量视为复数的实部和虚部，并返回极坐标形式下的模和角度

outer = torch.outer(torch.ones(3), torch.arange(1, 3)) # 计算两个张量的外积（第一个为列向量， 第二个为行向量，得到一个矩阵）

g = torch.arange(0, 6)
g_ = torch.cumsum(g, dim=0) # 在dim=0上累加

h = torch.multinomial(torch.Tensor([0, 10, 3, 0]), num_samples=4, replacement=True)   # 对tensor中的权重采样得到下标；num_samples是采样几次，replacement表示是否放回

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(complex_planar)
print(complex_planar_)
print(real_planar)
print(outer)
print(g)
print(g_)
print(h)
