import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch中的神经网络模块
import torch.optim as optim  # 导入PyTorch中的优化器模块
import torch.nn.functional as F  # 导入PyTorch中的函数式API

# 定义多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"  # 确保d_model可以被num_heads整除

        self.qkv_linear = nn.Linear(d_model, d_model * 3)  # 定义线性变换层，将输入映射到查询、键和值
        self.fc_out = nn.Linear(d_model, d_model)  # 定义输出层，将多头注意力的输出映射回原始维度

    def forward(self, x, mask=None):
        N, seq_length, _ = x.shape  # 获取输入的batch大小和序列长度
        qkv = self.qkv_linear(x).reshape(N, seq_length, 3, self.num_heads, self.head_dim)  # 计算查询、键和值，并重塑形状
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 重新排列维度以适应后续计算
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # 分离查询、键和值

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # 计算注意力得分
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # 应用掩码，将无效位置的得分设为负无穷

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)  # 计算注意力权重
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, self.d_model)  # 计算注意力加权输出并重塑形状
        out = self.fc_out(out)  # 通过输出层映射回原始维度
        return out  # 返回多头注意力的输出

# 定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 定义第一个线性变换层，将输入映射到隐藏层维度
        self.fc2 = nn.Linear(d_ff, d_model)  # 定义第二个线性变换层，将隐藏层输出映射回原始维度
        self.dropout = nn.Dropout(dropout)  # 定义Dropout层，用于防止过拟合

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 通过第一个线性层并应用ReLU激活函数
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)  # 通过第二个线性层
        return x  # 返回前馈神经网络的输出

# 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)  # 定义多头自注意力机制
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一个层归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二个层归一化层
        self.ff = FeedForward(d_model, d_ff, dropout)  # 定义前馈神经网络

    def forward(self, x, mask):
        attn_output = self.attention(x, mask)  # 通过多头自注意力机制
        x = self.norm1(attn_output + x)  # 残差连接并通过第一个层归一化层
        ff_output = self.ff(x)  # 通过前馈神经网络
        x = self.norm2(ff_output + x)  # 残差连接并通过第二个层归一化层
        return x  # 返回编码器层的输出

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)  # 定义多头自注意力机制
        self.norm1 = nn.LayerNorm(d_model)  # 定义第一个层归一化层
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)  # 定义编码器-解码器注意力机制
        self.norm2 = nn.LayerNorm(d_model)  # 定义第二个层归一化层
        self.norm3 = nn.LayerNorm(d_model)  # 定义第三个层归一化层
        self.ff = FeedForward(d_model, d_ff, dropout)  # 定义前馈神经网络

    def forward(self, x, enc_out, src_mask, trg_mask):
        attn_output = self.attention(x, trg_mask)  # 通过多头自注意力机制
        x = self.norm1(attn_output + x)  # 残差连接并通过第一个层归一化层
        enc_attn_output = self.encoder_attention(x, enc_out, src_mask)  # 通过编码器-解码器注意力机制
        x = self.norm2(enc_attn_output + x)  # 残差连接并通过第二个层归一化层
        ff_output = self.ff(x)  # 通过前馈神经网络
        x = self.norm3(ff_output + x)  # 残差连接并通过第三个层归一化层
        return x  # 返回解码器层的输出

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)  # 定义嵌入层，将输入序列映射到词向量
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # 创建多层编码器层
        self.dropout = nn.Dropout(dropout)  # 定义Dropout层

    def forward(self, x, mask):
        x = self.embedding(x)  # 输入序列经过嵌入层
        x = self.dropout(x)  # 应用Dropout
        for layer in self.layers:  # 逐层通过编码器层
            x = layer(x, mask)
        return x  # 返回编码器的输出

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, d_model)  # 定义嵌入层，将输入序列映射到词向量
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # 创建多层解码器层
        self.fc_out = nn.Linear(d_model, trg_vocab_size)  # 定义输出层，将解码器输出映射到目标词汇表大小
        self.dropout = nn.Dropout(dropout)  # 定义Dropout层

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.embedding(x)  # 输入序列经过嵌入层
        x = self.dropout(x)  # 应用Dropout
        for layer in self.layers:  # 逐层通过解码器层
            x = layer(x, enc_out, src_mask, trg_mask)
        x = self.fc_out(x)  # 通过输出层映射到目标词汇表大小
        return x  # 返回解码器的输出

# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)  # 定义编码器
        self.decoder = Decoder(trg_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)  # 定义解码器

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # 生成源序列掩码，忽略填充位置
        return src_mask  # 返回源序列掩码

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape  # 获取目标序列的batch大小和序列长度
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)  # 生成目标序列掩码，忽略未来位置
        return trg_mask  # 返回目标序列掩码

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  # 生成源序列掩码
        trg_mask = self.make_trg_mask(trg)  # 生成目标序列掩码
        enc_out = self.encoder(src, src_mask)  # 源序列经过编码器，生成编码器输出
        out = self.decoder(trg, enc_out, src_mask, trg_mask)  # 目标序列和编码器输出经过解码器，生成翻译结果
        return out  # 返回翻译结果

# 示例用法
if __name__ == "__main__":
    # 参数设置
    src_vocab_size = 5000  # 源语言词汇表大小
    trg_vocab_size = 5000  # 目标语言词汇表大小
    d_model = 512  # 模型维度
    num_heads = 8  # 多头注意力机制的头数
    num_layers = 6  # 编码器和解码器的层数
    d_ff = 2048  # 前馈神经网络的隐藏层大小
    dropout = 0.1  # Dropout概率

    # 创建Transformer模型实例
    model = Transformer(src_vocab_size, trg_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 创建模拟输入数据
    src = torch.randint(0, src_vocab_size, (32, 10))  # 模拟的源语言序列
    trg = torch.randint(0, trg_vocab_size, (32, 10))  # 模拟的目标语言序列

    # 前向传播，获取模型输出
    out = model(src, trg)

    # 打印输出形状
    print("Output shape:", out.shape)  # 应输出 (32, 10, trg_vocab_size)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 计算损失
    out = out.view(-1, trg_vocab_size)  # 将输出展平以计算损失
    trg = trg.view(-1)  # 将目标展平以计算损失
    loss = criterion(out, trg)

    # 打印损失值
    print("Loss:", loss.item())
