from typing import Any
import torch
from torch.nn import Linear, TransformerEncoderLayer
from torch.nn.functional import gumbel_softmax
from einops import rearrange, repeat

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, sequence_len, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positional_emb = torch.nn.Parameter(torch.randn(sequence_len, d_model))
        
    def forward(self, x, batch_size):
        positional_emb = repeat(self.positional_emb, 'seq emb -> b seq emb', b = batch_size)
        x += positional_emb
        return x

class GumbelSoftmaxBitRelaxation(torch.nn.Module):
    def __init__(self, temperature = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    def set_temperature(self, temperature):
        self.temperature = temperature
    
    def forward(self, x):
        x = gumbel_softmax(x, tau=self.temperature, hard=not(self.training), dim=-1)
        return x

class MaskingOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        x = input * mask
        return x
    @staticmethod
    def backward(ctx, grad_input):
        mask = ctx.saved_tensors
        dx = grad_input * mask
        d_mask = None
        return dx, d_mask

        
class ChannelTransformerBitLevelLargeVariable(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_feedback = dim_feedback
        self.tx_model = ChannelTransformerTransmitterBitLevel(n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.rx_model = ChannelTransformerReceiver(n_blocks=n_blocks, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, n_tx=n_tx, n_rx = n_rx, n_carrier=n_carrier, dim_feedback=dim_feedback)
        self.name = self.__class__.__name__ + '_' + str(dim_feedback)
        # self.neg_token = torch.nn.Parameter(torch.Tensor([-1]))
        
    def forward(self, x: torch.Tensor, no_bits):
        mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device)
        neg_mask = torch.ones([x.shape[0], self.dim_feedback]).to(x.device) * -1
        for i in range(len(no_bits)):
            mask[i, no_bits[i]:] = 0 
            neg_mask[i, :no_bits[i]] = 0
        x = self.tx_model(x)
        # x = MaskingOperation.apply(x, mask)
        x = x * mask
        x = x + neg_mask
        x = self.rx_model(x)
        return x
    def get_save_name(self):
        return self.name

    def set_temperature(self, temperatrue):
        self.tx_model.set_temperature(temperatrue)

class ChannelTransformerTransmitterBitLevel(torch.nn.Module):
    def __init__(self, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.output_adapter = Linear(in_features=n_tx * n_rx * 2, out_features=dim_feedback)
        self.output_activation = torch.nn.GELU()
        self.output_adapter_2 = Linear(in_features=self.n_carrier, out_features=32)
        
        self.output_adapter_3 = Linear(in_features=dim_feedback, out_features=dim_feedback)
        self.output_activation_2 = torch.nn.GELU()
        self.output_adapter_4 = Linear(in_features=32, out_features=2)
        
        self.bit_quantizer = GumbelSoftmaxBitRelaxation(temperature=5)
    
    def set_temperature(self, temperature):
        self.bit_quantizer.set_temperature(temperature)
    
    def forward(self, input_tensor):
        input_tensor = rearrange(input_tensor, 'b nrx ntx c complex -> b c (nrx ntx complex)')
        x = self.output_adapter(input_tensor)
        
        # x = x.mean(dim = 1)
        x = self.output_activation(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = self.output_adapter_2(x)
        
        x = rearrange(x, 'b feedback_dim c -> b c feedback_dim')
        x = self.output_adapter_3(x)
        x = self.output_activation_2(x)
        x = rearrange(x, 'b c feedback_dim -> b feedback_dim c')
        x = self.output_adapter_4(x)
        x = self.bit_quantizer(x)
        x = x[:,:,0]
        return x        



class ChannelTransformerReceiver(torch.nn.Module):
    def __init__(self, n_blocks, d_model, nhead, dim_feedforward, n_tx, n_rx, n_carrier, dim_feedback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.n_carrier = n_carrier
        
        self.encoders = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.encoders.append(
                TransformerEncoderLayer(
                    d_model = d_model,
                    nhead = nhead,
                    dim_feedforward = dim_feedforward,
                    activation = "gelu",
                    batch_first = True
                )
            )
               
        self.input_adapter = torch.nn.Sequential(
            Linear(in_features=dim_feedback, out_features=int((dim_feedback + d_model)//2)),
            torch.nn.BatchNorm1d(int((dim_feedback + d_model)//2)),
            torch.nn.GELU(),
            Linear(in_features=int((dim_feedback + d_model)//2), out_features=d_model),
        )
        self.output_adapters = torch.nn.ModuleList()
        for i in range(n_carrier):
            self.output_adapters.append(Linear(in_features=d_model, out_features=n_tx * n_rx * 2))
        self.output_activation = torch.nn.Tanh()
        self.positional_emb = PositionalEmbedding(d_model = d_model, sequence_len=n_carrier)
        
    def forward(self, input_tensor):
        input_tensor = self.input_adapter(input_tensor)
        
        batch_size, _ = input_tensor.shape
        input_tensor = repeat(input_tensor, 'b e -> b n e', n = self.n_carrier).clone()
        x = self.positional_emb(input_tensor, batch_size = batch_size)
        
        for enc in self.encoders:
            x = enc(x)
        
        out = []
        
        for i in range(self.n_carrier):
            out.append(self.output_adapters[i](x[:,i,:]))
        out = torch.stack(out, dim = 2) 
        
        out = rearrange(out, 'b (nrx ntx complex) ncarrier -> b nrx ntx ncarrier complex', ntx=self.n_tx, nrx=self.n_rx, ncarrier=self.n_carrier, complex = 2)
        # out = self.output_activation(out)
        
        return out
