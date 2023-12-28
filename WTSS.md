[vit pytorch源码](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

```python
class Attention(nn.Module):  # 多头注意力
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """
        dim: patch的维度
        dim_head: qkv的维度
        dropout: 神经元被丢弃的概率
        """
        super().__init__()
        inner_dim = dim_head *  heads  # 拼接各注意力头所得维度
        project_out = not (heads == 1 and dim_head == dim)  # 除非是单头注意力且前后维度一致，不然必须在拼接进行映射

        self.heads = heads
        self.scale = dim_head ** -0.5  # 放缩因子

        self.norm = nn.LayerNorm(dim)  # LN是对一个样本的所有特征进行归一化，所以要提供特征维度

        self.attend = nn.Softmax(dim = -1)  # 对最后一个维度进行归一化，即对一个query与各个key的相似度归一化
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 并行计算qkv

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 恢复输入维度（一般来说inner_dim == dim）
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # chunk作用是沿最后一个维度切三份，结果构成一个三元组赋值给qkv
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  #
        """
        b是batch_size大小
        n是序列长度
        h是注意力头的个数
        d即dim_head
        (h d)表示当前这一维度的大小等于hxd
        调整维度的目的是把n和d放在最后两维，从而在之后并行计算所有样本的所有head上的注意力图
        """

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 这里的两个四维矩阵相乘实质上是执行了bxh次的nxd矩阵乘dxn矩阵

        attn = self.attend(dots)  # softmax
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # 对输入进行加权和，这里同样是并行计算
        out = rearrange(out, 'b h n d -> b n (h d)')  # 并行计算后恢复原维度
        return self.to_out(out)  # MHSA最后的linear层
```

```python
class Transformer(nn.Module):  # transformer(decoder only)
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """
        dim: patch的维度
        depth: decoder包含的模块个数
        mlp_dim: 模块中的FeedForward网络（单隐藏层）的隐藏层神经元数量
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 添加depth个模块
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)  # 最后输出前进行LN
```

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227184652307.png" alt="image-20231227184652307" style="zoom: 67%;" />

```python
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):  
        """
        '*'表示后边的参数必须通过关键字传递而不能通过位置方式传递
        dim: patch的维度（不是patch的原始维度）
        dropout: decoder的dropout参数
        emb_dropout: 对patch进行embedding后所用dropout
        """ 
        super().__init__()
        image_height, image_width = pair(image_size)  # 返回元组(image_size, image_size)，下面同理
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # vit原论文的实验指出这两种使用decoder结果的方式其实没什么差别

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # 这里的(p1 p2 c)即为patch_dim（patch的原始维度）
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        """
        下面初始化（用标准正太分布初始化）了可训练的位置编码和cls
        第一个维度是batch_size维度
        第二个维度是序列长度
        第三个维度是patch维度
        """
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 将patch从原始维度映射到dim维
        b, n, _ = x.shape  # x形状为 b (h w) (p1 p2 c)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # 构建batch_size个cls
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]  # 直接加上位置编码
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 要么平均池化要么取cls编码结果

        x = self.to_latent(x)
        return self.mlp_head(x)
```

值得注意的细节：

vit使用gelu激活函数

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227101138494.png" alt="image-20231227101138494" style="zoom:33%;" />

vit使用layer norm

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227101307950.png" alt="image-20231227101307950" style="zoom:67%;" />

用LN的原因：transformer一开始用于NLP，输入序列的长度不确定，所以不适合做BN

此外，LN的位置其实有多种方案

transformer的输出是自回归的

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227103026757.png" alt="image-20231227103026757" style="zoom:33%;" />

position-wise

这指的是transformer的feed forward层和一般的mlp的区别：对每一个token单独做一次映射，权重共享

这个做法在处理序列输入时是非常常见和合理的，vit对每个patch的投影也是这么做的

transformer的自注意力计算

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227113742247.png" alt="image-20231227113742247" style="zoom: 50%;" />

因为输入数据一般是[batch_size, channels, height, weight]，样本数维度在最前面，所以推出来的是QK^TV

此外，softmax是对最后一个维度做归一化的    

多头注意力的计算

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227121124228.png" alt="image-20231227121124228" style="zoom:50%;" />

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227180415340.png" alt="image-20231227180415340" style="zoom: 50%;" />

其中的权重矩阵W代表Linear层，相比单头注意力，多头注意力多了两步映射操作（分别在进行注意力之前和之后）

注意每个head中将输入映射为dk=d_model/h维，这使得计算复杂度和单头注意力相似

transformer的decoder

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231227122454069.png" alt="image-20231227122454069" style="zoom: 50%;" />

decoder中有一部分没有使用自注意力，这里的K，V是编码器的输出，而Q是解码器的上一次的输出

CRATE源码

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231228224210406.png" alt="image-20231228224210406" style="zoom:67%;" />

ISTA

<img src="C:\Users\86134\AppData\Roaming\Typora\typora-user-images\image-20231228224229447.png" alt="image-20231228224229447" style="zoom:67%;" />

上式中需注意：

1. D*表示D的转置而不是伴随；
2. 这里的Z是一个列向量，而代码实现中每个样本对应的是行向量，所以要对输入x做列变换，即右乘参数矩阵（可使用linear层实现）

```python
class FeedForward(nn.Module):  # 实现ISTA
    def __init__(self, dim, hidden_dim, dropout = 0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))  
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):  # 输入Z^(l+1/2)
        # compute D^T * D * x 
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output  # 输出Z^(l+1)
```

MSSA





References

[Self Attention](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[ViT](https://arxiv.org/pdf/2010.11929.pdf)

[Survey of Transformers for Segmentation](https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197623X00104/1-s2.0-S0952197623008539/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDUaCXVzLWVhc3QtMSJIMEYCIQCgkOFQegKvk6j8fySXNEve4O3%2FTCjappPW9Cjj%2F8CPaAIhAN%2FgMeSYtpVDDuwUxM3puvFgyQI9smOuQLh3qSMB1rdwKrwFCL7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1Igzurio3%2FY8B9Dl%2BCKsqkAX1%2FpOy3qSwsZM%2BMKpsnJ6SGA9f%2B3X02yCLz8jUjbgBSwK5wsY3ft8eAKyhRRGkH3LrQ4koVE1%2B%2BETKgBAPUd15nYz3dqeV5H0XGrzhmt3hNAw87wN9LhCf6Ep19qswJ7TnjY4gOAJGuhFnxufl6tp2BubBv6OscIOZ2JEyjBWhViCok0FyJhqZWl0%2Fw8erTUvAgU6Jub6%2FtGDQqNO5Dgr0lZDEhXyOmL%2B9jtlLWfCXQzDcvSyUt6%2FUtQ7zzYHPHICKATrqoBf%2BbDgCvbn5ID7Md69LtkA6XQ4y0kei3w%2Bf%2FKsTZXRb4ql%2FaCT566%2B7zOgY1Pv0uw2eLTbhzas5ypDZ6P%2BHwRLo2E9W1JUDgoePvAiQwHT5kQ7teZRMCxFvRWvx32Z6wGKvxrldd9HaflGzFloX8W6HCA0THzM1P%2BKq5h4AAcFTWfiC97QUHuYP86mVz8CyTV5djL%2FWq7noXCFqquYVW0iefpeXubrkc2JzVBr2KwVGQT5o2CFz%2B1f3NfzIcO3lSb0fLQiERXRjaX819HbAeJBSUmvM1wy7jkQp3qLkv8k8w%2FlFJd4wIs2Ds6POC8gKuso1adOqgsSYjUSrhuhU4SGSvqxq71%2F%2FTNyCqSHBFoUNrzgxbiml6n0wroFH1%2BqOS59fim5h%2F56fqjyB5KirrvkWohyNmJXRIO%2BENBzD5VrMUTtoftedhhgTnGoaui%2BLuU%2Fm%2BQwUMhIGQYm6K5misJQu1Pc9MWrzghSLQR0aRnugSY3EH%2BBE2JOM7fV4ueqGj5wB2DxMG%2B0g0bj2EhYqWUvatbZwk6PUbCoCpLZfAqqUl%2FgTmemqzEMaZw41lPoH5t49U2VoqafU83wl9SomuXbRFhlH%2FsgokBzjYTDz4LWsBjqwAeQNsEJZx9ksvIWPKJVosGX5UTurFo%2FvSR76uC1KzxnJCW1OfaT4FvadQXaXLeoSa12LQT1Shioh04jFFY2ICIweIzfMiMbLfbs1W3f1y2zwly6fkJOEEKRCY6KwZiwr8K9QHSdBq0KVzhZRfmLGEkvItlOo5Zy%2BYpNFdmeP3BuAKraayvXhGCQ4CCm2RUAFQnbL2usTUHNLHTfGBwWE6OTzbR3pDtZB%2BMi3IFWvVtK4&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231228T132330Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7WPNJYMH%2F20231228%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=bc497eb55bfef1b9995df4d41af7bf91155b76aa12547f3cc5a331d3bc515fec&hash=3ba6b3284522660bae52a5a525e85e7a136295eeace69f8ac2517cd02f9f1725&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197623008539&tid=spdf-0cf89924-6637-49b6-947c-f729be653802&sid=8502f288692a5541896a5f20082ccfdb5909gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0e175a51050655045452&rr=83ca1e431c75f255&cc=tw)