import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from inspect import isfunction

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: The total time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process,
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        
        # ###########################################################
        self.alpha_t = cosine_schedule(timesteps).to(device)
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0).to(device)

    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        z = torch.randn(x.shape).to(x.device)
        t = torch.full((x.shape[0], ), t_index).to(device)
        this_alpha_t = extract(self.alpha_t, t, x.shape)
        this_alpha_bar_t = extract(self.alpha_bar_t, t, x.shape)
        eps_pred = self.model(x, torch.full((x.shape[0], ), t_index).to(device))
        x_0_pred = (x - torch.sqrt(1-this_alpha_bar_t)*eps_pred)/torch.sqrt(this_alpha_bar_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        if t_index == 0:
            return x_0_pred*(1-this_alpha_t)/(1-this_alpha_bar_t)
        else:
            prev_alpha_bar = extract(self.alpha_bar_t, t-1, x.shape)
            mu_bar_t = (torch.sqrt(this_alpha_t)*(1-prev_alpha_bar)/(1-this_alpha_bar_t))*x + (torch.sqrt(prev_alpha_bar)*(1-this_alpha_t)/(1-this_alpha_bar_t))*x_0_pred
            var_t = ((1-prev_alpha_bar)/(1-this_alpha_bar_t))*((1-this_alpha_t))
            return mu_bar_t + torch.sqrt(var_t)*z
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generted image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        for t_index in range(self.num_timesteps-1, -1, -1):
            img = self.p_sample(img, None, t_index)
        img = torch.clamp(img, -1, 1)
        img = unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the p_sample_loop function ####
        img = torch.randn((batch_size, self.channels, self.image_size, self.image_size)).to(device)
        img = self.p_sample_loop(img)
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise=None):
        """
        Samples from the noise distribution at time t. Apply alpha interpolation between x_start and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from. If None, noise will be sampled.
        Returns:
            The sampled image.
        """
        ###### TODO: Implement the q_sample function #######
        
        if noise is None:
            noise = torch.randn(x_0.shape).to(x_0.device)
        this_alpha_bar_t = extract(self.alpha_bar_t, t, x_0.shape)
        x_t = torch.sqrt(this_alpha_bar_t) * x_0 + torch.sqrt(1 - this_alpha_bar_t)*noise
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use. If None, noise will be sampled.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        if noise is None:
            noise = torch.randn(x_0.shape)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t)
        loss_func = nn.L1Loss()
        loss = loss_func(noise_pred, noise)
        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        t = torch.randint(0, self.num_timesteps, (b, )).to(device)
        return self.p_losses(x_0, t, noise)
