import einops
import numpy as np
import torch
from torch.func import functional_call, grad, vmap


from typing import Callable, List, Optional, Tuple, Union
from warnings import warn

from abc import ABC, abstractmethod
from tqdm import tqdm


class NeuralODE(torch.nn.Module, ABC):
    @classmethod
    @abstractmethod
    def t_unnoised(cls):
        pass

    @classmethod
    @abstractmethod
    def t_noised(cls):
        pass

    @classmethod
    @abstractmethod
    def latent_cond(cls, n: int):
        pass

def handle_oom_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError("Out of memory error caught, please decrease the vmap chunk_size.") from e
    return wrapper


class RectFlow(NeuralODE):
    def __init__(self, d: int, approximator: torch.nn.Module, device: str = None):
        super().__init__()
        self.d = d
        self.ode_approx = approximator
        self.device = device
    @classmethod
    def t_unnoised(cls):
        return 1.0
    @classmethod
    def t_noised(cls):
        return 0.0
    @classmethod
    def latent_cond(cls, n: int):
        return torch.distributions.Independent(torch.distributions.Normal(torch.tensor(n*[0.]), torch.tensor(n*[1.])), 1)
    @classmethod
    def loss(
        cls,
        net: torch.nn.Module,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor, 
        t: torch.Tensor) -> torch.FloatTensor:
        xt = t*x1 + (1-t)*x0
        return ((x1 - x0 - net(xt, t))**2).sum(-1)
    def __call__(self, t: float, x: torch.Tensor, *args) -> torch.Tensor:
            is_ndarray = False
            if isinstance(x, np.ndarray):
                is_ndarray = True
                x = torch.from_numpy(x).float()
            is_flat = False
            if len(x.shape) == 1:
                is_flat = True
                bz = len(x) // self.d
                x = x.view(bz, self.d)
            elif len(x.shape) == 2:
                bz = x.shape[0]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            if self.device is not None:
                x = x.to(self.device)
                t = torch.tensor(t, device=self.device)
            t_tensor = t.float().expand(bz,1)
            dxdt = self.ode_approx(t_tensor, x, *args)
            if is_ndarray:
                dxdt = dxdt.cpu().numpy()
            if is_flat:
                dxdt = dxdt.flatten()
            return dxdt
    def noise_data(self, data : torch.FloatTensor, noises: torch.FloatTensor, times: torch.FloatTensor):
        noisy_sample = times[..., None] * data + (1-times[..., None])*noises
        return noisy_sample


class LikelihoodSolver(torch.nn.Module):
    def __init__(self,
            ode: torch.nn.Module,
            d : int,
            chunk_size : Optional[int] = None,
            num_positional_args : int = 0
        ):
        super().__init__()
        self.ode = ode
        self.d = d
        ft_compute_grad = grad(self.get_trace, argnums=2, has_aux=True)
        self.ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, *[0 for _ in range(num_positional_args)]), chunk_size=chunk_size)
        self.params = {k: v.detach() for k, v in self.ode.named_parameters()} 
        self.buffers_ = {k: v.detach() for k, v in self.ode.named_buffers()} 

    def __call__(self, x, t):
        return self.ode(t, x)

    def get_trace(self, params, buffers, xs, t, e, *args):
        f_val = functional_call(self.ode, (params, buffers), (t, xs, *args))
        return einops.einsum(f_val, e, '... d, ... d->'), f_val

    def skilling_hutchinson_trace_vmap_f(
        self,
        t: Union[float, torch.Tensor],
        x : Union[torch.Tensor, np.ndarray],
        e: torch.Tensor,
        returns_f: bool = False,
        *args,
    ):
        # n, bz, ..., d = e.shape
        n = e.shape[0]
        x_shape = e.shape[1:]
        dimension_less_shape = x_shape[:-1]
        device = next(self.parameters()).device
        if returns_f:
            assert len(x.shape) == 1
            x, logp_diff = einops.unpack(x, [x_shape, dimension_less_shape], '*')
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(t, float):
            t = torch.tensor([t]).float()

        x = x.to(device)
        t = t.to(device)
        e = e.to(device)
   
        x = x[None].expand(e.shape)
        x = einops.rearrange(x, '... d -> (...) 1 d')
        t = t.repeat(n, *dimension_less_shape)
        t = t.flatten()
        e_ = einops.rearrange(e, '... d -> (...) 1 d')
        if args:
            _args = []
            for arg in args:
                if arg.ndim == 3 and arg.shape[:-1] == dimension_less_shape:
                    _arg = arg[None].repeat_interleave(n, dim=0)
                    _arg = einops.rearrange(_arg, '... d -> (...) 1 d')
                    _args.append(_arg)
                else:
                    raise NotImplementedError
            args = _args
        vecJacProds, f_vals = self.ft_compute_sample_grad(self.params, self.buffers_, x, t, e_, *args)
        vecJacProds = vecJacProds.reshape(e.shape)
        trace_ests = einops.einsum(vecJacProds,e, '... d, ... d -> ...')
        trace_ests = -trace_ests.mean(0)
        f_vals = f_vals.view(e.shape)
        f_vals = f_vals[0]
        ret, ps =  einops.pack([f_vals, -trace_ests], '*') if returns_f else trace_ests
        return ret.cpu()
    @handle_oom_error
    def diffusion_logp(
        self,
        x_init: torch.FloatTensor,
        t_unnoised: float,
        t_noised: float,
        t_eval: Optional[List[float]] = None,
        coordinate_transform:  Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = None,
        latent_cond: Optional[torch.distributions.Distribution] = torch.distributions.Independent(torch.distributions.Normal(0, 1.), -1),
        n_runs: Optional[int] = 1000,
        skilling_hutchinson_noise: Optional[torch.Tensor] = None,
        skilling_hutchinson_vector_type: str = 'rademacher',
        solver: str = 'scipy',
        backward = True,
        generator: Optional[torch.Generator] = None,
        show_progress = False,
        **solver_kwargs
    ):
        '''
        Estimates the log likelihood of the diffusion process at the given time points.
        
        Parameters:
        x_init: torch.FloatTensor
            if backward, is a point in the unnoised space, otherwise noised
        t_unnoised: float
            time unnoised
        t_noised: float
            time noised
        t_eval: Optional[List[float]]
            evaluate likelihood at these noise levels...
        coordinate_transform:  Optional[Callable[[torch.FloatTensor], torch.FloatTensor]]
            data normalization if backward, otherwise denormalization
        latent_cond: Optional[torch.distributions.Distribution]
            diffusion latent noise distribution if unset, we use standard normal
        n_runs: Optional[int]
            number of runs used for estimating skilling_hutchinson_trace,
        skilling_hutchinson_noise: Optional[torch.Tensor]
            skilling noise, if set ignores n_runs
        skilling_hutchinson_vector_type: str
            type of skilling noise, either 'rademacher' or 'gaussian'
        solver: str
            solver to use, either 'scipy' or 'euler'
        backward: bool
            if True we solve the backward diffusion, otherwise forward
        generator: Optional[torch.Generator]
            random number generator
        **solver_kwargs
            additional arguments to the solver
            
        Returns:
        logp_est: np.ndarray
            estimated log likelihood
        x_ret: np.ndarray
            estimated diffusion path
        eval_points: np.ndarray
            time evaluation points
        
        Example:
        
        >>> likelihood_solver = LikelihoodSolver(ode=ode, d=2)
        >>> x_init = torch.randn(2, 2)
        >>> logp_est, x_ret, eval_points = likelihood_solver.diffusion_logp(x_init, t_unnoised=0.0, t_noised=1.0, t_eval=[1.0, 0.5, 0.0], backward=True)
        >>> eval_points[0] == 1.0
        >>> x_init == x_ret[0]
        
        
        '''
        if coordinate_transform and t_eval:
            warn("t_eval and coordinate_transform are both set, noised samples will be coordinate transformed and their likelihoods corrected.")
        with torch.no_grad():
            x0 = x_init
            if backward:
                if coordinate_transform:
                    x0 = coordinate_transform(x0)
                start_int = t_unnoised
                end_int = t_noised
            else:
                start_int = t_noised
                end_int = t_unnoised
            dimension_less_shape = x0.shape[:-1]
            zeros = np.zeros((*dimension_less_shape,))
            
            init_con, ps = einops.pack([x0.cpu().numpy(), zeros], '*')
            device = next(self.ode.parameters()).device
            if skilling_hutchinson_noise is None:
                skilling_hutchinson_noise = self.skilling_hutchinson_noise(n_runs, dimension_less_shape, skilling_hutchinson_vector_type, generator=generator)
            skilling_hutchinson_noise = skilling_hutchinson_noise.to(device)
            args = (skilling_hutchinson_noise,True)
            solver_kwargs_copy = solver_kwargs.copy()
            if 'args' in solver_kwargs:
                args = args + solver_kwargs_copy.pop('args')
            try:
                if backward and t_eval and t_eval[-1] != t_noised:
                    t_eval = t_eval + [t_noised]
                if solver == 'scipy':
                    import scipy.integrate
                    ret = scipy.integrate.solve_ivp(
                        self.skilling_hutchinson_trace_vmap_f,
                        t_span=(start_int,end_int),
                        y0=init_con,
                        t_eval=t_eval,
                        args = args,
                        **solver_kwargs_copy
                    )
                    sol = ret.y
                    eval_points = ret.t
                elif solver == 'euler':
                    assert ('step_size' in solver_kwargs) != ('n_steps' in solver_kwargs), 'Please provide either step_size or n_steps'
                    if 'step_size' in solver_kwargs:
                        step_size = solver_kwargs['step_size']
                        raise NotImplementedError('The code below, does not not ensure that the interval is "divisable with the step"')
                        assert abs((end_int - start_int) % step_size) < 1e-9
                        eval_points = np.arange(start_int, end_int, step_size)
                    elif 'n_steps' in solver_kwargs:
                        eval_points = np.linspace(start_int, end_int, solver_kwargs['n_steps']+1, endpoint=True)
                    if t_eval: 
                        match_eval_points = [np.where(np.isclose(eval_points, te))[0][0] for te in t_eval]
                        assert len(match_eval_points) == len(t_eval), f'The eval_points {list(t_eval)} should be in the interval, {list(eval_points)}'
                    sol = np.empty((len(init_con), len(eval_points)))
                    sol[:, 0] = init_con
                    for i, eval_point in enumerate(tqdm(eval_points[:-1], desc='Solving ODE euler', disable=not show_progress)):
                        x = sol[:,i]
                        sol[:,i+1] = x + (eval_points[i+1]-eval_points[i])*self.skilling_hutchinson_trace_vmap_f(eval_point, x, *args).numpy()
                    if t_eval:
                        sol = sol[:,match_eval_points]
                        eval_points = t_eval
                else:
                    raise NotImplementedError
            except torch.cuda.OutOfMemoryError as e:
                print('Out of memory error caught, decrease the vmap chunk_size.')
                raise e

            x_solved, logp_diff = einops.unpack(sol.T, ps, 't *')
            x_solved = torch.as_tensor(x_solved, dtype=torch.float32)
            
            if backward:
                # NOTE: 3rd element subtracts the part of the integral from the start (data) to the t_eval noised point.
                x_latent = x_solved[-1]
                logpz = latent_cond.log_prob(x_latent).numpy()
                logp_est = (logpz + logp_diff[-1:] - logp_diff)
                correction = self.coordinate_correction(coordinate_transform, coordinates=x_init, backward=True).numpy()
                x_ret = x_solved
            else:
                # NOTE: since we solve forward, we need to change the sign.
                x_latent = x_init
                logpz = latent_cond.log_prob(x_latent).numpy()
                logp_est = (logpz - logp_diff)
            
                correction = self.coordinate_correction(coordinate_transform, coordinates=x_solved, backward=False).numpy()
                if coordinate_transform is None:
                    x_ret = x_solved
                else: 
                    x_ret = coordinate_transform(x_solved)

            logp_est += correction
        return logp_est, x_ret, eval_points
    
    @classmethod
    def coordinate_correction(
        cls,
        coordinate_transform : Callable[[torch.Tensor], torch.Tensor],
        coordinates : torch.Tensor,
        backward = True
    ):
        '''Returns the log likelihood change of the correction for the coordinate transformation.'''
        # https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function
        if coordinate_transform is None:
            correction = torch.tensor(0.0, device=coordinates.device)
        else:
            def correction_fn(coordinate, coordinate_transform):
                jac = torch.func.jacrev(coordinate_transform)(coordinate)
                return torch.linalg.slogdet(jac)

            def multi_vmap(f, n_dims):
                in_dims = (0, None)
                for _ in n_dims:
                    f = vmap(f, in_dims)
                return f
            correction_vmap = multi_vmap(correction_fn, torch.arange(len(coordinates.shape)-1))
            sign, log_abs_det = correction_vmap(coordinates, coordinate_transform)
            if not backward:
                # NOTE: see the inverse function theorem https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Inverse
                log_abs_det = -log_abs_det
            correction = log_abs_det
        return correction

    def skilling_hutchinson_noise(self, n_runs: int, shape : Union[int, List[int]], noise_type: str = 'rademacher', generator: Optional[torch.Generator]=None):
        if isinstance(shape, int):
            shape = [shape]
        if noise_type == 'gaussian':
            skilling_noise = torch.randn(n_runs, *shape, self.d, generator=generator)
        elif noise_type == 'rademacher':
            skilling_noise = torch.randint(2, size=(n_runs, *shape, self.d), generator=generator).float() * 2 - 1
        else:
            raise NotImplementedError
        return skilling_noise

class NeuralODELL:
    def __init__(
        self,
        d : int,
        ode: NeuralODE,
        chunk_size : int,
        n_runs_skillet : int,
        n_samples: int,
        normalize: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = None,
        denormalize: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = None,
        solver : Tuple[str, dict] = ('scipy', {'atol':1e-3, 'rtol':1e-3}),
        noise_type : Optional[str] = None,
        num_positional_args : int = 0
    ):
        assert (denormalize is None) == (normalize is None), "Either both or neither of normalize and denormalize should be provided"
        self.likelihood_solver = LikelihoodSolver(ode=ode, d=d, chunk_size=chunk_size, num_positional_args=num_positional_args)
        self.ode = ode
        if noise_type is not None:
            self.skillet_noise = self.likelihood_solver.skilling_hutchinson_noise(n_runs=n_runs_skillet, shape=n_samples, noise_type=noise_type)
        else:
            self.skillet_noise = None
        self.n_runs_skillet = n_runs_skillet
        self.solver, self.solver_kwargs = solver
        self.normalize = normalize
        self.denormalize = denormalize
        self.latent_cond_ = self.ode.latent_cond(d)
        torch.manual_seed(0)
        self.latent_samples = self.latent_cond_.sample((n_samples,))
    
    def solve_backward(self, samples : torch.FloatTensor, t_eval : Optional[List[float]] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        n = samples.shape[0]
        assert self.skillet_noise is None or n == self.skillet_noise.shape[1], "Number of samples should match number of noise samples"
        return self.likelihood_solver.diffusion_logp(
                x_init=samples,
                t_unnoised=self.ode.t_unnoised(),
                t_noised=self.ode.t_noised(),
                t_eval=t_eval,
                coordinate_transform=self.normalize,
                latent_cond=self.latent_cond_,
                n_runs=self.n_runs_skillet,
                skilling_hutchinson_noise=self.skillet_noise,
                solver=self.solver,
                **self.solver_kwargs
            )

    def solve_forward(self, t_eval : Optional[List[float]] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.likelihood_solver.diffusion_logp(
            x_init=self.latent_samples,
            t_unnoised=self.ode.t_unnoised(),
            t_noised=self.ode.t_noised(),
            t_eval=t_eval,
            coordinate_transform=self.denormalize,
            latent_cond=self.latent_cond_,
            n_runs=self.n_runs_skillet,
            skilling_hutchinson_noise=self.skillet_noise,
            solver=self.solver,
            backward=False,
            **self.solver_kwargs
        )
