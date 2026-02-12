import os
from pathlib import Path
from typing import Dict, List, Optional
import librosa
from music2latent import EncoderDecoder
import numpy as np
import soundfile as sf
import math
import torch
import torchaudio
from jsonargparse import CLI
import csv

from .diffusion_models import RectFlowsTransformerModel
from dataclasses import dataclass, field, asdict

DIM_DATA = 64
PERIODE_MUSIC_2_LATENT = 4096 / 44100


def get_model(
    dim_in,
    dim_out,
    model_type,
    max_seq_len=4600,
    generator=None,
    transformer_wrapper_hparams: dict = dict(emb_dropout=0.1),
    causal_transformer_hparams=dict(
        dim=256,
        attn_flash=True,
        attn_dropout=0.1,
        ff_dropout=0.1,
        rotary_pos_emb=True,
        depth=12,
        heads=8,
    ),
    n_cls=0,
    noise_context=None,
):
    if model_type == "LTM-rectflows":
        model = RectFlowsTransformerModel(
            data_dim=dim_in,
            max_seq_len=max_seq_len,
            transformer_wrapper_hparams=transformer_wrapper_hparams,
            causal_transformer_hparams=causal_transformer_hparams,
            pos_enc="fourier",
            generator=generator,
            noise_context=noise_context,
            n_cls=n_cls,
        )
    else:
        raise ValueError("Invalid model type specified. Choose 'Standard' or 'STM'.")

    return model


def from_ckpt(ckpt):
    """
    Load a pre-trained ContinuousTransformerModel from a checkpoint file.

    Args:
        ckpt (str): Path to the checkpoint file.

    Returns:
        ContinuousTransformerModel: An instance of the ContinuousTransformerModel
        loaded with the state dictionary from the checkpoint and set to evaluation mode.
    """
    ckpt_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_state_dict = ckpt_dict["model_state_dict"]
    # dim_out = model_state_dict['model.project_out.weight'].shape[0]
    dim_out = DIM_DATA
    max_seq_length = ckpt_dict["max_seq_length"]
    model_type = ckpt_dict["model_type"]
    model_kwargs = ckpt_dict.get("model_kwargs", dict())
    model = get_model(
        dim_in=DIM_DATA,
        dim_out=dim_out,
        model_type=model_type,
        max_seq_len=max_seq_length,
        **model_kwargs,
    )
    model.load_state_dict(model_state_dict)
    model = model.eval()
    return model


def compute_non_silence(audio, silence_extractor: str, fixed_sr: int):
    """
    Compute the non-silence regions of an audio signal.

    Args:
        audio (np.ndarray): The audio signal as a NumPy array.
        silence_extractor (str): Method to compute non-silence regions ('music' or 'voice').
        fixed_sr (int): The fixed sample rate of the audio signal.

    Returns:
        np.ndarray: An array of non-silence regions, where each row represents
                    [start_index, end_index] of a non-silent segment.
    """
    if silence_extractor == "music":
        # Use librosa's split function to detect non-silent regions in music
        non_silence = librosa.effects.split(audio)
    elif silence_extractor == "voice":
        # Use torchaudio's Voice Activity Detection (VAD) for voice signals
        vad = torchaudio.transforms.Vad(sample_rate=fixed_sr)
        in_aud = torch.as_tensor(audio)
        # Calculate the silence at the beginning and end of the audio
        heading_silence = audio.shape[-1] - vad(in_aud).shape[-1]
        trailing_silence = vad(in_aud.flip([-1])).shape[-1]
        # Represent the non-silence region as a single range
        non_silence = np.array([[heading_silence, trailing_silence]])
    else:
        # Raise an error if the silence extractor type is not implemented
        raise NotImplementedError
    return non_silence


@dataclass
class IntegrationParams:
    """Integration and solver parameters for likelihood evaluation.

    Attributes:
        n_runs (int): Number of Monte Carlo samples used for the Skilling-Hutchingson divergence estimator (https://arxiv.org/pdf/1810.01367) in the "instantaneous change of variables formula" (https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf). Setting this value low (e.g., setting it to `1`, which is feasible in some cases) speeds up performance.
        solver (str): ODE solver selection — either 'euler' (internal Euler integrator) or 'scipy'
            (use scipy.integrate.solve_ivp).
        solver_kwargs (Dict): Keyword args for the chosen solver. For 'euler' set 'n_steps' as the number of euler steps;
            for 'scipy' these are passed to scipy.integrate.solve_ivp (e.g., method, atol, rtol).
    """

    n_runs: int = 4
    solver: str = "euler"
    solver_kwargs: Dict = field(default_factory=lambda: {"n_steps": 100})


class ICCalcHelper:
    def __init__(
        self,
        device,
        ckpt: Optional[str] = None,
    ):
        """
        Helper class for calculating Information Content (IC).

        Args:
            device (str): The device to use for computation (e.g., 'cuda' or 'cpu').
            ckpt (Optional[str]): Path to the model checkpoint. If None, download or use cached pretrained model.
        """
        if ckpt is None:
            filepath = os.path.abspath(__file__)
            lib_root = os.path.dirname(filepath)
            if not os.path.exists(lib_root + "/models/pa_rectflow.pt"):
                # NOTE: if the latent model does not exist, download it from Hugging Face
                print("Downloading model...")
                from huggingface_hub import hf_hub_download

                os.makedirs(lib_root + "/models", exist_ok=True)
                _ = hf_hub_download(
                    repo_id="CPJKU/pa-audioic",
                    filename="pa_rectflow.pt",
                    cache_dir=lib_root + "/models",
                    local_dir=lib_root + "/models",
                )
                print("Model was downloaded successfully!")
            ckpt = lib_root + "/models/pa_rectflow.pt"

        self.model = from_ckpt(ckpt).to(device)
        self.device = device
        self.encdec = EncoderDecoder(
            load_path_inference=None, device=torch.device(device)
        )

    @torch.no_grad
    def ic(
        self,
        heading_nan_pads: List[int],
        trailing_nan_pads: List[int],
        latents: List[torch.Tensor],
        # ode_kwargs: Dict,
        noise_levels: List[float] = [0.0],
        monte_carlo_samples: Optional[int] = None,
        noise_from_expection: bool = True,
        integration_params: IntegrationParams = IntegrationParams(),
        vmap_chunk_size: int = 196608
    ):
        """
        Calculate the negative log-likelihood (NLL) for the given latent representations.

        Args:
            heading_nan_pad (List[int]): Number of NaN padding values at the start of the sequence.
            trailing_nan_pad (List[int]): Number of NaN padding values at the end of the sequence.
            latents (List[torch.Tensor]): Latent representations.
            noise_levels (List[float]): Noise levels / times at which to evaluate the IC (list of floats between 0 and 1, where 0 is clean and 1 is fully noised).
            monte_carlo_samples (Optional[int]): Number of Monte Carlo samples used when calculating IC with noised data. If None, uses expected value of noise process. Otherwise, performs Monte Carlo estimate of noise process.
            noise_from_expection (bool): Whether to compute IC from the expected noise process or use the probability flow ODE.
            integration_params (IntegrationParams): Integration and solver parameters for likelihood evaluation. See IntegrationParams.
            vmap_chunk_size (int): Vectorization chunk size used when calculating IC for multiple time-steps in parallel. Set lower if you run into out-of-memory issues.

        Returns:
            List[np.ndarray]: NLL values with NaN padding applied.
        """
        len_latents = [lat.shape[0] for lat in latents]
        latents = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True)
        if self.model.__class__.__name__ == "RectFlowsTransformerModel":
            # NOTE: model was trained with a definition of rectified flow where t=1.0 is unnoised and t=0.0 is fully noised.
            # To keep compatible with conventional definitions of noise levels, the noise levels are inverted here.
            ode_kwargs = asdict(integration_params)
            noise_levels = [1.0 - level for level in noise_levels]
            ode_kwargs["t_eval"] = noise_levels
        if noise_from_expection:
            logp_est, noised_mag, t_eval_backward = self.model.logp_from_expectation(
                latents, ode_kwargs, monte_carlo_samples=monte_carlo_samples, vmap_chunk_size=vmap_chunk_size
            )
        else:
            logp_est, noised_mag, t_eval_backward = self.model.ll_new(
                latents, **ode_kwargs, vmap_chunk_size=vmap_chunk_size
            )
        nlls = -torch.as_tensor(logp_est, dtype=torch.float32, device=latents.device)
        nlls = nlls.to("cpu")
        ret = []
        for nll, l, heading_nan_pad, trailing_nan_pad in zip(
            nlls.transpose(0, 1), len_latents, heading_nan_pads, trailing_nan_pads
        ):
            ret.append(
                np.pad(
                    nll[:, :l],
                    ((0, 0), (heading_nan_pad, trailing_nan_pad)),
                    mode="constant",
                    constant_values=np.nan,
                )
            )
        return ret

    @torch.no_grad
    def entr(
        self,
        heading_nan_pads: List[int],
        trailing_nan_pads: List[int],
        latents: List[torch.Tensor],
        noise_levels: List[float] = [0.0],
        monte_carlo_samples: int = 4,
        integration_params: IntegrationParams = IntegrationParams(),
        vmap_chunk_size: int = 196608,
    ):
        """
        Calculate the entropy for the given latent representations.

        Args:
            heading_nan_pad (List[int]): Number of NaN padding values at the start of the sequence.
            trailing_nan_pad (List[int]): Number of NaN padding values at the end of the sequence.
            latents (List[torch.Tensor]): Latent representations.
            noise_levels (List[float]): Noise levels / times at which to evaluate the entropy (list of floats between 0 and 1, where 0 is clean and 1 is fully noised).
            monte_carlo_samples (Optional[int]): Number of Monte Carlo samples used when calculating unbiased entropy estimate.
            integration_params (IntegrationParams): Integration and solver parameters for likelihood evaluation. See IntegrationParams.
            vmap_chunk_size (int): Vectorization chunk size used when calculating entropy for multiple time-steps in parallel. Set lower if you run into out-of-memory issues.
        Returns:
            List[np.ndarray]: Entropy values with NaN padding applied.
        """
        len_latents = [lat.shape[0] for lat in latents]
        latents = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True)
        if self.model.__class__.__name__ == "RectFlowsTransformerModel":
            # NOTE: model was trained with a definition of rectified flow where t=1.0 is unnoised and t=0.0 is fully noised.
            # To keep compatible with conventional definitions of noise levels, the noise levels are inverted here.
            ode_kwargs = asdict(integration_params)
            noise_levels = [1.0 - level for level in noise_levels]
            ode_kwargs["t_eval"] = noise_levels
        logp_est, samples, t_eval_forward = self.model.pred_new(
            latents, **ode_kwargs, vmap_chunk_size=vmap_chunk_size, samples_pr_context=monte_carlo_samples
        )
        nlls = -torch.as_tensor(logp_est, dtype=torch.float32, device=latents.device)
        entrs = nlls.mean(dim=2)
        entrs = entrs.to("cpu")
        ret = []
        for entr, l, heading_nan_pad, trailing_nan_pad in zip(
            entrs.transpose(0, 1), len_latents, heading_nan_pads, trailing_nan_pads
        ):
            ret.append(
                np.pad(
                    entr[:, :l],
                    ((0, 0), (heading_nan_pad, trailing_nan_pad)),
                    mode="constant",
                    constant_values=np.nan,
                )
            )
        return ret
    def encode_m2l(self, audio_file: str | np.ndarray, silence_extractor):
        """
        Encodes an audio file or array into latent representations, while handling silence and padding.

        Args:
            audio_file (str | np.ndarray): Path to the audio file or a NumPy array of audio samples.
            silence_extractor (str): Method to compute non-silence regions ('music' or 'voice').

        Returns:
            Tuple[int, int, int, torch.Tensor]:
                - heading_nan_pad: Number of NaN padding values at the start.
                - trailing_nan_padding: Number of NaN padding values at the end.
                - len_wo_pad: Length of the sequence without padding.
                - latents_padded: Padded latent representations.
        """
        if isinstance(audio_file, str):
            audio, rate = sf.read(audio_file, dtype="float32")
            if rate != 44100:
                audio_tensor = torch.from_numpy(audio)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=rate, new_freq=44100
                )
                audio = resampler(audio_tensor)
        else:
            audio = audio_file
            rate = 44100

        # If the audio has multiple channels, convert it to mono by averaging
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)
        audio_samples = len(audio)
        non_silence = compute_non_silence(audio, silence_extractor, rate)
        # Silence at the beginning
        heading_silence = non_silence[0][0]
        # Silence at the end
        trailing_silence = non_silence[-1][1]

        # Calculate the number of NaN padding values for the start and end
        heading_nan_pad = math.floor(heading_silence / (rate * PERIODE_MUSIC_2_LATENT))
        trailing_nan_padding = math.ceil(
            (audio_samples - trailing_silence) / (rate * PERIODE_MUSIC_2_LATENT)
        )
        audio = audio[heading_silence:trailing_silence]
        latents = self.encdec.encode(audio)
        # Adjust dimensions for processing
        latents = latents[0].float().permute(1, 0)
        return heading_nan_pad, trailing_nan_padding, latents
    def frames_to_time(self, n_frames: int) -> np.ndarray:
        """
        Convert frame indices to time in seconds.

        Args:
            n_frames (int): Number of frames.
        Returns:
            np.ndarray: Array of time values in seconds corresponding to each frame.
        """
        return (np.arange(n_frames) + 1) * PERIODE_MUSIC_2_LATENT

def calc_ic(
    audio_files: List[str],
    noise_levels: List[float] = [0.0],
    audio_type: str = "music",
    output_dir: str = "./",
    device: str = "cuda",
    monte_carlo_samples: Optional[int] = None,
    noise_from_expection: bool = True,
    integration_params: IntegrationParams = IntegrationParams(),
    bz: int = 1,
    vmap_chunk_size: int = 196608
):
    """Calculate the Information Content (IC) for each audio file and store the results in CSV files.

    Args:
        audio_files (List[str]): A list of audio file paths to process.
        noise_levels (List[float], optional): Noise levels / times at which to evaluate the IC (list of floats between 0 and 1, where 0 is clean and 1 is fully noised).
        audio_type (str, optional): The type of audio being processed ('music' or 'voice'). Used for replacing heading and trailing silence with NaN values.
        output_dir (str, optional): The directory where output files will be saved.
        device (str, optional): The device to use for computation ('cuda' or 'cpu').
        monte_carlo_samples (Optional[int]): Number of Monte Carlo samples used when calculating IC with noised data. If None, uses expected value of noise process. Otherwise, performs Monte Carlo estimate of noise process
        noise_from_expection (bool): Whether to compute IC from the expected noise process or use the probability flow ODE.
        integration_params (IntegrationParams, optional): Integration and solver parameters for likelihood evaluation.
        bz (int, optional): Batch size for processing audio files. Setting this >1 can speed up computation if audio files are short and approximately uniform in length. 
        vmap_chunk_size (int): Vectorization chunk size used when calculating IC for multiple time-steps in parallel. Set lower if you run into out-of-memory issues.
    """
    ic_calc_helper = ICCalcHelper(device=device)
    data_loader = torch.utils.data.DataLoader(
        audio_files, batch_size=bz, shuffle=False, drop_last=False
    )
    for audio_files in data_loader:
        heading_nan_pad, trailing_nan_padding, latents = zip(
            *[
                ic_calc_helper.encode_m2l(audio_file, silence_extractor=audio_type)
                for audio_file in audio_files
            ]
        )
        nll = ic_calc_helper.ic(
            heading_nan_pad,
            trailing_nan_padding,
            latents,
            noise_levels=noise_levels,
            integration_params=integration_params,
            noise_from_expection=noise_from_expection,
            monte_carlo_samples=monte_carlo_samples,
            vmap_chunk_size=vmap_chunk_size,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for audio_file, nll in zip(audio_files, nll):
            time = ic_calc_helper.frames_to_time(nll.shape[1])
            csv_filename = output_dir.joinpath(f"{Path(audio_file).stem}.csv")
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time", *[f"IC_{level}" for level in noise_levels]])
                writer.writerows(zip(time, *nll))


if __name__ == "__main__":
    CLI(calc_ic, as_positional=True)
