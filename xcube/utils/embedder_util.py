import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, include_input=True, input_dims=3, max_freq_log2=10, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        super().__init__()
        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            out_dim += d
            
        max_freq = max_freq_log2
        N_freqs = num_freqs
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for _ in periodic_fns:
                out_dim += d
        
        self.include_input = include_input
        self.freq_bands = freq_bands
        self.periodic_fns = periodic_fns
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def forward(self, inputs):
        output_list = [inputs]
        for fn in self.embed_fns:
            output_list.append(fn(inputs))
            
        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                output_list.append(p_fn(inputs * freq))
        
        return torch.cat(output_list, -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embedder_obj = Embedder(max_freq_log2=multires-1, num_freqs=multires, input_dims=input_dims)
    return embedder_obj, embedder_obj.out_dim