
from typing import Dict, List, Optional, Tuple
from warnings import warn
import einops
from .diffusion import RectFlow
import numpy as np
import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Decoder
from .diffusion import LikelihoodSolver

def zero_init(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
def init(module):
    torch.nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        torch.nn.init.constant_(module.bias, 0.)
    return module

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16, generator=None):
        super().__init__()
        self.register_buffer('freqs', 2*torch.pi*torch.randn(num_channels // 2, generator=generator) * scale)

    def forward(self, x : torch.Tensor):
        x = einops.einsum(x, self.freqs.to(x.dtype),'..., b -> ... b')
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x

class AdaNorm(torch.nn.Module):
    # https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L19-L20
    def __init__(self, dim: Tuple, cond_dim: Tuple , bias=True):
        super().__init__()
        self.norm = self.norm = torch.nn.LayerNorm(normalized_shape=dim, elementwise_affine=False, bias=False)
        self.scale_proj = zero_init(torch.nn.Linear(cond_dim, dim))
        if bias:
            self.shift_proj = zero_init(torch.nn.Linear(cond_dim, dim))
    def forward(self, x, cond):
        normed = self.norm(x)
        scale = self.scale_proj(cond)
        if hasattr(self, 'shift_proj'):
            shift = self.shift_proj(cond)
        else:
            shift = torch.zeros_like(scale)
        return normed * (1 + scale) + shift
class Feedforward(torch.nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mlp_mult)
        dim_out = dim

        self.activation = torch.nn.SiLU()
        self.to_mlp = init(torch.nn.Linear(dim, inner_dim, bias=False))
        self.to_out = zero_init(torch.nn.Linear(inner_dim//2, dim_out, bias=False))
        self.do = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_mlp(x)
        x1,x2 = x.chunk(2, dim=-1)
        x = self.activation(x1) * x2
        x = self.do(x)
        x = self.to_out(x)
        return x
class MLPAdaNorm(torch.nn.Module):
    def __init__(self, dim, mlp_mult=4, cond_dim=None, dropout=0., bias=False):
        super().__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, dropout=dropout)
        self.norm = AdaNorm(dim, cond_dim, bias)

    def forward(self, x, cond):
        inp = x
        x = self.norm(x, cond)
        x = self.ff(x)
        return x+inp
class DiffusionMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim=None, dim=512, num_layers=12, mlp_mult=1, adanorm_bias=False):
        super().__init__()
        self.emb_proj = torch.nn.Sequential(
            init(torch.nn.Linear(cond_dim, dim)),
            torch.nn.SiLU(),
            init(torch.nn.Linear(dim, dim)),
            torch.nn.SiLU(),
            init(torch.nn.Linear(dim, dim)),
            torch.nn.SiLU()
        )
        self.time_emb = FourierEmbedding(num_channels=cond_dim)
        self.linear_input = init(torch.nn.Linear(input_dim, dim))
        self.norm_output = AdaNorm(dim, cond_dim=cond_dim)
        self.linear_output = init(torch.nn.Linear(dim, output_dim))
        self.layers = torch.nn.ModuleList([
            MLPAdaNorm(dim, mlp_mult=mlp_mult, cond_dim=cond_dim, bias=adanorm_bias) for _ in range(num_layers) 
        ])
    def forward(self, x : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        t = self.emb_proj(self.time_emb((t * 2.)-1.))
        x = self.linear_input(x)
        for layer in self.layers:
            x = layer(x, t)
        x = self.norm_output(x, t)
        x = self.linear_output(x)
        return x    

class DiffusionMLPContext(DiffusionMLP):
    def __init__(self, input_dim, output_dim, cond_dim=None, dim=512, num_layers=12, mlp_mult=1, adanorm_bias=False):
        super().__init__(input_dim, output_dim, cond_dim, dim, num_layers, mlp_mult, adanorm_bias)
    def forward(self, times : torch.Tensor, noisy_sample : torch.Tensor, context_state : torch.Tensor) -> torch.Tensor:
        diff_in = torch.cat([noisy_sample, context_state], dim=-1)
        v = super().forward(diff_in, times)
        return v

class ODETransformerModel(nn.Module):
    def __init__(
        self,
        data_dim,
        max_seq_len,
        transformer_wrapper_hparams: dict = dict(emb_dropout = 0.1),
        causal_transformer_hparams : dict = dict(dim = 256, attn_flash=True, attn_dropout = 0.1,ff_dropout = 0.1, rotary_pos_emb=True),
        pos_enc='fourier',
        generator=None,
        noise_context=None,
        n_cls=0,
    ):
        super().__init__()
        

        self.n_cls = n_cls
        self.noise_context = noise_context
        dim = causal_transformer_hparams.get('dim')
        self.backbone = ContinuousTransformerWrapper(
            dim_in=data_dim,
            dim_out=None,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                **causal_transformer_hparams
            ),
            **transformer_wrapper_hparams
        )
        self.generator = generator
        self.diff_mlp = DiffusionMLPContext(input_dim=dim+data_dim, output_dim=data_dim, cond_dim=dim, dim=dim, num_layers=8, mlp_mult=4, adanorm_bias=False)
    def cls_backbone_foward(self, x):
        cls_tokens = torch.randn(x.shape[0], 1, x.shape[2], generator=torch.Generator().manual_seed(99)).to(x.device)
        cls_tokens = cls_tokens.expand(-1, self.n_cls, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        mask = torch.ones(x.size(0), x.size(1)).bool().to(x.device)
        start_idx = max(self.n_cls-1,0)
        context_state = self.backbone(x, mask=mask)[:,start_idx:]
        return context_state


    def ll_new(
        self,
        x : torch.Tensor,
        n_runs : int = 128,
        vmap_chunk_size : int = 196608,
        solver = 'scipy',
        solver_kwargs = {},
        t_eval : Optional[List[float]] = None,
        t_unnoised : Optional[float] = None
    ):
        if self.n_cls > 0:
            raise NotImplementedError('Need to update to use cls tokens')
        if self.noise_context:
            raise NotImplementedError('Need to add the noise during inference')
        if t_unnoised is None:
            t_unnoised = self.ode.t_unnoised()
        
        noise_type = 'rademacher'
        bz, T, dim_data = x.shape
        mask = torch.ones(x.size(0), x.size(1)).bool().to(x.device)
        targets = x[:, 1:]
        context_state =  self.backbone(x, mask=mask)[:,:-1].detach()
        solver_kwargs = dict(solver_kwargs, args=(context_state,))
        likelihood_solver = LikelihoodSolver(ode=self.ode, d=dim_data, chunk_size=vmap_chunk_size, num_positional_args=1)
        sh_noise = likelihood_solver.skilling_hutchinson_noise(n_runs, targets.shape[:-1], noise_type, generator=self.generator)
        # NOTE: SOLVE backward
        logp_est, noise, t_eval_backward = likelihood_solver.diffusion_logp(
            x_init=targets.cpu(),
            t_unnoised=t_unnoised,
            t_noised=self.ode.t_noised(),
            latent_cond=self.ode.latent_cond(dim_data),
            skilling_hutchinson_noise=sh_noise,
            solver=solver,
            t_eval=t_eval,
            **solver_kwargs
        )
        return logp_est, noise, t_eval_backward
    @torch.no_grad()
    def pred_new(
        self,
        x : torch.Tensor,
        n_runs : int = 128,
        vmap_chunk_size : int = 196608,
        solver = 'scipy',
        solver_kwargs = {},
        t_eval : Optional[List[float]] = None,
        samples_pr_context: int = 1,
        teacher_forcing : bool =True,
    ):
        noise_type = 'rademacher'
        bz, T, dim_data = x.shape
        if self.noise_context:
            warn('Train/Inference discrepancy, no context noise is added.')
            pass
        context_state = self.cls_backbone_foward(x)
        if teacher_forcing:
            context_state =  context_state[:, :-1]
        else:
            context_state = context_state[:, -1:]
        context_state = context_state[None].expand(samples_pr_context, *context_state.shape)
        context_state = einops.rearrange(context_state, 'n b t d -> (n b) t d')
        solver_kwargs = dict(solver_kwargs, args=(context_state,))
        x_init = self.ode.latent_cond(dim_data).sample(context_state.shape[:-1])
        likelihood_solver = LikelihoodSolver(ode=self.ode, d=dim_data, chunk_size=vmap_chunk_size, num_positional_args=1)
        
        sh_noise = likelihood_solver.skilling_hutchinson_noise(n_runs, x_init.shape[:-1], noise_type, generator=self.generator)
        logp_est_prediction, prediction, t_eval_forward = likelihood_solver.diffusion_logp(
            x_init=x_init,
            t_unnoised=self.ode.t_unnoised(),
            t_noised=self.ode.t_noised(),
            latent_cond=self.ode.latent_cond(dim_data),
            skilling_hutchinson_noise=sh_noise,
            solver=solver,
            backward=False,
            t_eval=t_eval,
            **solver_kwargs
        )
        logp_est_prediction = einops.rearrange(logp_est_prediction, 'trace (n b) t -> trace b n t', n=samples_pr_context)
        prediction = einops.rearrange(prediction, 'trace (n b) t d -> trace b n t d', n=samples_pr_context)
        return logp_est_prediction, prediction, t_eval_forward
    def loss(self, x : torch.Tensor):
        raise NotImplementedError
    def logp_from_expectation(
        self,
        mag : torch.Tensor,
        ode_kwargs : Dict,
        monte_carlo_samples : Optional[int] = None,
        vmap_chunk_size : int = 196608,
    ):
        if self.n_cls > 0:
            raise NotImplementedError('Need to update to use cls tokens')
        if self.noise_context:
            raise NotImplementedError('Need to add the noise during inference')
        logp_est = []
        ode_kwargs_copy = dict(**ode_kwargs)
        t_eval_backward = ode_kwargs_copy.pop('t_eval')
        device = mag.device
        expand_dim = 1 if monte_carlo_samples is None else monte_carlo_samples
        mag = mag[None].expand(expand_dim, *mag.shape)
        mag = einops.rearrange(mag, 'n b t d -> (n b) t d')
        if monte_carlo_samples is None:
            noise = torch.zeros_like(mag)
        else:
            noise = torch.randn_like(mag)
        for nle in t_eval_backward:
            times = torch.full(mag.shape[:-1], fill_value=nle, device=device)
            noised_expected = self.ode.noise_data(mag, noise, times)
            noised_expected = einops.rearrange(noised_expected, '(n b) t d -> n b t d', n=expand_dim)
            noised_expected = noised_expected.mean(dim=0)
            logp_det_mid_process, _, _ = self.ll_new(
                x=noised_expected,
                t_unnoised=nle, 
                vmap_chunk_size=vmap_chunk_size,
                **ode_kwargs_copy,
            )
            logp_est.append(logp_det_mid_process[0,:])
        logp_est = np.stack(logp_est, 0)
        return logp_est, noised_expected.cpu().numpy(), t_eval_backward

class RectFlowsTransformerModel(ODETransformerModel):
    def __init__(self, data_dim, max_seq_len,transformer_wrapper_hparams, causal_transformer_hparams, pos_enc='fourier', generator= None, noise_context  : Optional[str]  = None, n_cls=0):
        super().__init__(data_dim, max_seq_len, transformer_wrapper_hparams, causal_transformer_hparams, pos_enc, generator, noise_context, n_cls)
        self.ode = RectFlow(d=data_dim, approximator=self.diff_mlp)