import torch

print(torch.__version__)           # 输出 2.11.0+cu124 这种，+cu124代表cuda12.4版本
print(torch.cuda.is_available())   # 返回True才算GPU可用；False=GPU没跑起来
print(torch.cuda.get_device_name(0)) # 输出 RTX 5060