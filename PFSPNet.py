import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PFSP import getMeanCmax

default_config = dict(
    n_layers = 6,  # number of Encoder (Decoder) Layer: N 
    n_heads = 8,  # number of heads in Multi-Head Attention: h
    d_model = 512,  # Embedding Siz
    d_ff = 2048,  # FeedForward dimension
    d_k = 64, 
    d_v = 64,  # dimension of K(=Q), V
    m_max = 20,  # The max number of machine
)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, inputs_Q, inputs_K, inputs_V):
        """
        Args:
            inputs_Q: [batch_size, q_n, d_model]
            inputs_K: [batch_size, k_n, d_model]
            inputs_V: [batch_size, v_n, d_model]

        Returns:
            outputs: [batch_size, q_n, d_model]
            attn: [batch_size, n_heads, q_n, k_n]
        """

        batch_size = inputs_Q.size(0)
        # (B, S, D) -Linear-> (B, S, D_new) -Split-> (B, S, H, W) -Trans-> (B, H, S, W)
        Q = self.W_Q(inputs_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Q: [batch_size, n_heads, q_n, d_k]
        K = self.W_K(inputs_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, k_n, d_k]
        V = self.W_V(inputs_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # V: [batch_size, n_heads, v_n, d_v]

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores : [batch_size, n_heads, ([q_n, d_k] * [d_k, k_n]) -> q_n, k_n]
        attn = F.softmax(scores, dim=-1)
        # attn : [batch_size, n_heads, q_n, k_n]
        context = torch.matmul(attn, V)
        # context : [batch_size, n_heads, ([q_n, k_n] * [v_n, d_v]) -> q_n, d_v]
        # k_n == v_n

        # Concat
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_v
        )  # context: [batch_size, q_n, n_heads * d_v]

        # Linear
        outputs = self.fc(context)  # [batch_size, q_n, d_model]

        return outputs, attn
    
class MultiHeadAttention_Res_LN(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.LN = nn.LayerNorm(d_model)
        
    def forward(self, input_Q, input_K, input_V):
        """
        Args:
            input_Q: [batch_size, q_n, d_model]
            input_K: [batch_size, k_n, d_model]
            input_V: [batch_size, v_n, d_model]
            
        Returns:
            outputs: [batch_size, q_n, d_model]
            attn: [batch_size, n_heads, q_n, k_n]
        """
        
        residual = input_Q
        outputs, attn = self.MHA(input_Q, input_K, input_V)
        return self.LN(outputs + residual), attn
    
class PoswiseFeedForwardNet_Res_LN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=True), nn.ReLU(), nn.Linear(d_ff, d_model, bias=True))
        self.LN = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, n, d_model]

        Returns:
            outputs: [batch_size, n, d_model]
        """

        residual = inputs
        outputs = self.fc(inputs)
        return self.LN(outputs + residual)
        # [batch_size, n, d_model]
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention_Res_LN(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.ffn = PoswiseFeedForwardNet_Res_LN(d_model=d_model, d_ff=d_ff)
        
    def forward(self, inputs):
        """
        inputs: [batch_size, n, d_model]
        """

        # First Part
        outputs, _ = self.self_attn(inputs, inputs, inputs)
        # outputs: [batch_size, n, d_model], attn: [batch_size, n_heads, n, n]

        # Second Part
        outputs = self.ffn(outputs)
        # outputs: [batch_size, n, d_model]

        return outputs
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, n_layers):
        super().__init__()
        self.Layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff) for _ in range(n_layers)])
        
    def forward(self, inputs):
        """
        inputs: [batch_size, n+1, d_model]
        """
        
        outputs = inputs
        
        for layer in self.Layers:
            outputs = layer(outputs)
        # outputs: [batch_size, n+1, d_model]
        
        return outputs
    
class Critic(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff)
        self.projection = nn.Linear(d_model, 1, bias=False)
        
    def forward(self, inputs):
        """
        inputs: [batch_size, n+1, d_model]
        """

        outputs = self.layer(inputs)
        # [batch_size, n+1, d_model]
        
        outputs = outputs.sum(dim=1)
        # [batch_size, d_model]
        
        outputs = self.projection(outputs).view(-1)
        # [batch_size]
        
        return outputs
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.encoder_attn = MultiHeadAttention_Res_LN(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.self_attn = MultiHeadAttention_Res_LN(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.ffn = PoswiseFeedForwardNet_Res_LN(d_model=d_model, d_ff=d_ff)
        
    def forward(self, inputs_decoder, outputs_encoder):
        """
        inputs_decoder: [batch_size, n, d_model]
        outputs_encoder: [batch_size, n+1, model]
        """

        # First Part
        outputs, _ = self.self_attn(inputs_decoder, inputs_decoder, inputs_decoder)
        # outputs: [batch_size, n, d_model], attn: [batch_size, n_heads, n, n]

        # Second Part
        outputs, _ = self.encoder_attn(outputs, outputs_encoder, outputs_encoder)
        # outputs: [batch_size, n, d_model], attn: [batch_size, n_heads, n, n]

        # Third Part
        outputs = self.ffn(outputs)
        # outputs: [batch_size, n, d_model]

        return outputs
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, n_layers):
        super().__init__()
        self.Layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff) for _ in range(n_layers)])
        self.projection = nn.Linear(d_model, 1, bias=False)
        
    def forward(self, inputs_decoder, outputs_encoder):
        """
        inputs_decoder: [batch_size, n, d_model]
        outputs_encoder: [batch_size, n+1, model]
        """
        
        outputs = inputs_decoder
        
        for layer in self.Layers:
            outputs = layer(outputs, outputs_encoder)
        # outputs: [batch_size, n, d_model]
        
        outputs = self.projection(outputs).view(outputs.shape[0], outputs.shape[1])
        # outputs: [batch_size, n]
        
        return outputs
    
class PFSPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_decoder = nn.Linear(config['m_max'], config['d_model'])
        self.emb_encoder = nn.Linear(config['m_max'], config['d_model'])
        self.emb_n = nn.Linear(1, config['d_model'])
        self.emb_m = nn.Linear(1, config['d_model'])
        
        self.encoder = Encoder(config['d_model'], config['n_heads'], config['d_k'], config['d_v'], config['d_ff'], config['n_layers'])
        self.decoder = Decoder(config['d_model'], config['n_heads'], config['d_k'], config['d_v'], config['d_ff'], config['n_layers'])
        self.crtic = Critic(config['d_model'], config['n_heads'], config['d_k'], config['d_v'], config['d_ff'])
        
    def forward(self, inputs: torch.tensor, state: torch.tensor)->torch.tensor:
        """
        Args:
            inputs (torch.tensor): [batch_size, n, m_max] 带选择的下个 Job 集合 
            state (torch.tensor): [batch_size, m_max] 上一个 Job 各个工序的完成时间

        Returns:
            torch.tensor: [batch_size, n, probs]
        """

        n = torch.ones_like(inputs[:, :, 0]).sum(1).view(-1, 1)
        n = self.emb_n(n).unsqueeze(1)
        # [batch_size, 1, d_model]

        m = (inputs.sum(1) > 0).sum(1).view(-1, 1).type_as(inputs)
        m = self.emb_m(m).unsqueeze(1)
        # [batch_size, 1, d_model]
        
        initTime = state[:, 0]
        state = state - initTime.view(-1, 1)  # 使得 state 之间差异更小，这并不影响预测下一 Job
        state = state.unsqueeze(1)
        # [batch_size, m_max] -> [batch_size, 1, d_model]
        
        encoder_inputs = torch.cat([state, inputs], dim=1)  # 由于 state 数据特性和其余 inputs 不一致，所以这里没有额外编码
        encoder_inputs = self.emb_encoder(encoder_inputs)
        encoder_inputs = encoder_inputs + n + m
        # [batch_size, n+1, d_model]

        decoder_inputs = self.emb_decoder(inputs)
        decoder_inputs = decoder_inputs + m + n
        # [batch_size, n, m_max] -> [batch_size, n, d_model]

        # encoder
        encoder_outputs = self.encoder(encoder_inputs)
        # [batch_size, n+1, d_model]
        
        # decoder
        decoder_outputs = self.decoder(decoder_inputs, encoder_inputs)
        decoder_outputs = F.softmax(decoder_outputs, dim=-1)
        # [batch_size, n]
        
        # critic
        baselines = self.crtic(encoder_outputs) + initTime
        # [batch_size]
        
        return decoder_outputs, baselines
        
def gen_data(dataset_size, n, m, m_max=20):
    P = (
        0.6 * torch.rand(dataset_size, n, m_max)
        + 0.2 * torch.rand(dataset_size, 1, m_max)
        + 0.2 * torch.rand(dataset_size, n, 1)
    )
    P[:, :, m:] = 0

    state = 1.2 * torch.rand(dataset_size, m_max)
    state[:, m:] = 0
    for i in range(1, m_max):
        state[:, i] += state[:, i - 1]
    return P, state

class PFSPDataset(Dataset):
    def __init__(self, dataset_size, n, m, m_max=20, USE_CUDA=True, PRETrainCritic=False):
        self.PRETrainCritic = PRETrainCritic
        self.dataset_size = dataset_size
        self.P, self.state = gen_data(dataset_size, n, m, m_max)
        
        if torch.cuda.is_available() and USE_CUDA:
            self.P, self.state = self.P.cuda(), self.state.cuda()
            
        if PRETrainCritic is True:
            self.label = getMeanCmax(self.P, self.state)

    def __getitem__(self, index):
        if self.PRETrainCritic is True:
            return self.P[index], self.state[index], self.label[index]
        else:
            return self.P[index], self.state[index]

    def __len__(self):
        return self.dataset_size
    
def PFSPDataLoader(dataset_size, batch_size, n, m, m_max=20, USE_CUDA=True, PRETrainCritic=False):
    return DataLoader(PFSPDataset(dataset_size, n, m, m_max=m_max, USE_CUDA=USE_CUDA, PRETrainCritic=PRETrainCritic), batch_size, shuffle=True)