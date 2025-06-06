import torch
from torch import nn
from torch.amp import autocast
import torch.nn.functional as F

from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple
from einops import reduce


# constantsƒgaus
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size = (9, 9),
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        device = "cpu"
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, mask = None, context = None, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
     
    def condition_mean(self, cond_fn, mean,variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # this fixes a bug in the official OpenAI implementation:
        # https://github.com/openai/guided-diffusion/issues/51 (see point 1)
        # use the predicted mean for the previous timestep to compute gradient
        gradient = cond_fn(mean, t, **guidance_kwargs)
        new_mean = (
            mean.float() + variance * gradient.float()
        )
        print("gradient: ",(variance * gradient.float()).mean())
        return new_mean
    
    def compute_log_p(self, x, mu, variance, model_log_variance, mask, ddpo=True):
        '''
        Compute log p(zs | zt) for a Gaussian distribution.
        '''
        # Ensure sigma is positive to avoid numerical issues
        epsilon = 1e-6
        variance = torch.max(variance, torch.tensor(epsilon, device=variance.device))

        delta = x.detach() if ddpo else x

        log_p = -0.5 * ((delta - mu) ** 2 / variance + model_log_variance * (~mask) + torch.tensor(2. * torch.pi, device=x.device) * (~mask))
        log_p = log_p.mean(dim=tuple(range(1, log_p.ndim)))  # sum over spatial dimensions （1,2,3）
        
        return log_p

    def p_sample(self, x, t: int, mask = None, context = None, x_self_cond = None, prev_sample = None, cond_fn=None, is_training=True, guidance_kwargs=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, mask = mask, context = context, x_self_cond = x_self_cond, clip_denoised = True
        )
        
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, batched_times, guidance_kwargs)
        
        model_mean = mask * context + (~mask) * model_mean
        variance = (~mask) * variance
        model_log_variance = (~mask) * model_log_variance + mask * (-float(1e22))
        
        if is_training:
            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            img = model_mean + (0.5 * model_log_variance).exp() * noise
        else:
            img = model_mean
        
        # compute logp
        if prev_sample != None :
            log_p = self.compute_log_p(prev_sample, model_mean, variance, model_log_variance, mask, True)
        else:
            log_p = self.compute_log_p(img, model_mean, variance, model_log_variance, mask, False)
        
        return img, x_start, log_p, model_mean, (0.5 * model_log_variance).exp()

    def p_sample_loop(self, shape, mask = None, context = None, return_all_timesteps = False, cond_fn=None, is_training=True, guidance_kwargs=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(context.shape, device = device)

        x_start = None

        mask = mask if mask is not None else torch.zeros_like(img, dtype=torch.bool)
        context = context if context is not None else torch.zeros_like(img, device=device)
        img = torch.where(mask, context, img)
        
        latents = [img]
        logps = []
        timesteps = []
        mus = []
        sigmas = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = context if self.self_condition else None
            img, x_start, logp, mu, sigma = self.p_sample(img, t=t, mask=mask, context=context, x_self_cond=self_cond, prev_sample=None,
                                                          cond_fn=cond_fn, is_training=is_training, guidance_kwargs=guidance_kwargs)
            latents.append(img)
            logps.append(logp)
            timesteps.append(t)
            mus.append(mu)
            sigmas.append(sigma)

        ret = img if not return_all_timesteps else torch.stack(latents, dim = 1)

        ret = self.unnormalize(ret)
        
        return ret, latents, logps, timesteps, mus, sigmas
    
    def p_sample_ddim(self, x, time: int, time_next: int, mask = None, context = None, time_cond = None, x_self_cond = None, prev_sample = None, is_training=True):
        pred_noise, x_start, *_ = self.model_predictions(x, time_cond, x_self_cond, clip_x_start = True)
        
        model_mean, variance, model_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = time_cond)
        
        model_mean = mask * context + (~mask) * model_mean
        variance = (~mask) * variance
        model_log_variance = (~mask) * model_log_variance + mask * (-float(1e22))
        
        if time_next < 0:
            img = x_start
        else:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            if is_training:
                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            else:
                img = x_start * alpha_next.sqrt() + c * pred_noise
        
        # compute logp
        if prev_sample != None :
            log_p = self.compute_log_p(prev_sample, model_mean, variance, model_log_variance, mask, True)
        else:
            log_p = self.compute_log_p(img, model_mean, variance, model_log_variance, mask, False)
        
        return img, x_start, log_p, model_mean, (0.5 * model_log_variance).exp()

    def ddim_sample(self, shape, mask = None, context = None, return_all_timesteps = False, cond_fn=None, is_training=True, guidance_kwargs=None):
        batch, device, total_timesteps, sampling_timesteps, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        x_start = None

        mask = mask if mask is not None else torch.zeros_like(img, dtype=torch.bool)
        context = context if context is not None else torch.zeros_like(img, device=device)
        img = torch.where(mask, context, img)
        
        latents = [img]
        logps = []
        timesteps = []
        mus = []
        sigmas = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            latents.append(img)
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            
            img, x_start, logp, mu, sigma = self.p_sample_ddim(img, time, time_next, mask=mask, context=context, time_cond = time_cond, x_self_cond=self_cond, prev_sample=None, is_training=is_training)
            
            latents.append(img)
            logps.append(logp)
            timesteps.append(time)
            mus.append(mu)
            sigmas.append(sigma)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret, latents, logps, timesteps, mus, sigmas

    @torch.no_grad()
    def sample(self, batch_size = 16, mask = None, context = None, return_all_timesteps = False, cond_fn=None, is_training=True, guidance_kwargs=None):
        h, w, channels = self.image_size[0], self.image_size[1], self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), mask = mask, context = context, return_all_timesteps = return_all_timesteps,
                         cond_fn=cond_fn, is_training=is_training, guidance_kwargs=guidance_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, context, mask, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        x = mask * context + (~mask) * x

        # predict and take gradient step
        model_out = self.model(x, t, context)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out * ~mask, target * ~mask, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, context, mask, *args, **kwargs):
        b, c, h, w, device, h_img_size, w_img_size = *img.shape, img.device, self.image_size[0], self.image_size[1]
        assert h == h_img_size and w == w_img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t, context, mask, *args, **kwargs)