# PAAudioIC
PAAudioIC provides tools for calculating the *information content* (IC) as a proxy for human-perceived surprise when listening to music.  This repo is the official implementation of [*"Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders"*, Bjare et al., NeurIPS - AI for Music Workshop, 2025](https://openreview.net/forum?id=rXUKO0ysUy).

PAAudioIC includes a command-line tool and Python classes for calculating IC using a rectified flow (diffusion) model.


## Installation
You can install the package using pip with or without the extra dependencies required for the demo.

### Install the package for general use:
```bash
pip install git+https://github.com/CPJKU/pa-audioic
```

### Install the package with demo dependencies:
```bash
git clone https://github.com/CPJKU/pa_audioic.git
cd pa-audioic
pip install ".[demo]"
```

## Usage
### Using the AudioIC programmatically
The [`demo.ipynb`](./demo.ipynb) notebook demonstrates how to use the library programmatically to calculate and visualize the IC of audio files.

### Running the `audioic` Command-Line Tool
#### Basic usage
The [`audioic`](./pa_audioic/audioic.py) command-line tool allows you to compute the *information content* (IC) of audio files.
To use it, specify the audio files you want to process and provide an output directory where the results will be saved as CSV files:

```bash
python -m pa_audioic.audioic --output_dir OUTPUT_DIR --device "cpu" "['AUDIO_FILE_1','AUDIO_FILE_2',...]"
```

Replace `AUDIO_FILE_1`, `AUDIO_FILE_2`, etc., with the paths to your audio files, and `OUTPUT_DIR` with the directory where you want the output files to be stored.

To get hierarchical perceptual IC estimates, run:
```bash
python -m pa_audioic.audioic --output_dir OUTPUT_DIR --noise_levels "[NOISE_LEVEL_1,NOISE_LEVEL_2,...]" --device "cpu" "['AUDIO_FILE_1','AUDIO_FILE_2',...]"
```
Replace `NOISE_LEVEL_1`, `NOISE_LEVEL_2`, etc., with float numbers in `[0.0, 1.0]`, where, `0.0` corresponds to estimating IC of all information in the audio signal, `1.0` corresponds to using none of the audio signal information and intermediate values (e.g `0.6`) corresponds to removing parts of the information found to be less perceptually relevant (see paper for more details).  

To run the tool on a GPU (default), specify the `--device` argument as `"cuda"`:

```bash
CUDA_VISIBLE_DEVICES=DEVICE_ID python -m pa_audioic.audioic --output_dir OUTPUT_DIR --device "cuda" "['AUDIO_FILE_1','AUDIO_FILE_2',...]"
```
Replace `DEVICE_ID` with a cuda device id.

#### CSV output format
Running `audioic` with parameters
```bash
python -m pa_audioic.audioic --noise_levels "[0.0,0.3,0.5]"  "['AUDIO_FILE_1']"
```
will result in a an audio file following the following format.
```csv
Time,IC_0.0,IC_0.3,IC_0.5
0.09287981859410431,nan,nan,nan
0.18575963718820862,nan,nan,nan
0.2786394557823129,nan,nan,nan
0.37151927437641724,55.07291,38.968643,39.15696
0.46439909297052157,74.925934,55.580364,48.62501
...
7.987664399092971,45.44641,43.173058,35.923733
8.080544217687075,39.387936,28.39015,33.411766
8.17342403628118,36.639862,28.572035,32.627956
8.266303854875284,nan,nan,nan
8.359183673469389,nan,nan,nan
```
**Time** — Reports the time in seconds (reported as the timestamp of the last sample decoded from the predicted music2latent frame).
**IC_NOISE_LEVEL** — IC reported at NOISE_LEVEL. `nan` is reported where the model detects heading or trailing silence in the audio file.

#### Advanced usage
To list the full program arguments, run:
```
python -m pa_audioic.audioic --help

usage: audioic.py [-h] [--config CONFIG] [--print_config[=flags]] [--noise_levels NOISE_LEVELS] [--audio_type AUDIO_TYPE] [--output_dir OUTPUT_DIR] [--device DEVICE] [--monte_carlo_samples MONTE_CARLO_SAMPLES]
                  [--noise_from_expection {true,false}] [--integration_params CONFIG] [--integration_params.n_runs N_RUNS] [--integration_params.solver SOLVER] [--integration_params.solver_kwargs SOLVER_KWARGS] [--bz BZ]
                  [--vmap_chunk_size VMAP_CHUNK_SIZE]
                  audio_files

Calculate the Information Content (IC) for each audio file and store the results in CSV files.

positional arguments:
  audio_files           A list of audio file paths to process. (required, type: List[str])

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: skip_default, skip_null.
  --noise_levels NOISE_LEVELS, --noise_levels+ NOISE_LEVELS
                        Noise levels / times at which to evaluate the IC (list of floats between 0 and 1, where 0 is clean and 1 is fully noised). (type: List[float], default: [0.0])
  --audio_type AUDIO_TYPE
                        The type of audio being processed ('music' or 'voice'). Used for replacing heading and trailing silence with NaN values. (type: str, default: music)
  --output_dir OUTPUT_DIR
                        The directory where output files will be saved. (type: str, default: ./)
  --device DEVICE       The device to use for computation ('cuda' or 'cpu'). (type: str, default: cuda)
  --monte_carlo_samples MONTE_CARLO_SAMPLES
                        Number of Monte Carlo samples used when calculating IC with noised data. If None, uses expected value of noise process. Otherwise, performs Monte Carlo estimate of noise process (type: Optional[int], default:
                        null)
  --noise_from_expection {true,false}
                        Whether to compute IC from the expected noise process or use the probability flow ODE. (type: bool, default: True)
  --bz BZ               Batch size for processing audio files. Setting this >1 can speed up computation if audio files are short and approximately uniform in length. (type: int, default: 1)
  --vmap_chunk_size VMAP_CHUNK_SIZE
                        Vectorization chunk size used when calculating IC for multiple time-steps in parallel. Set lower if you run into out-of-memory issues. (type: int, default: 196608)

Integration and solver parameters for likelihood evaluation:
  --integration_params CONFIG
                        Path to a configuration file.
  --integration_params.n_runs N_RUNS
                        Number of random samples used for divergence Skilling-Hutchinson trace estimation. (type: int, default: 4)
  --integration_params.solver SOLVER
                        ODE solver selection — either 'euler' (internal Euler integrator) or 'scipy' (use scipy.integrate.solve_ivp). (type: str, default: euler)
  --integration_params.solver_kwargs SOLVER_KWARGS
                        Keyword args for the chosen solver. For 'euler' set 'n_steps' as the number of euler steps; for 'scipy' these are passed to scipy.integrate.solve_ivp (e.g., method, atol, rtol). (type: Dict, default:
                        {'n_steps': 100})
```
The CLI application is built with [jsonargparse](https://jsonargparse.readthedocs.io/en/v4.43.0) and supports setting all arguments either via configuration files or by parsing them as `JSON` strings, as described in the documentation.

##### Noising-related arguments
As described in the paper, IC can be estimated in a way that filters less perceptual information by adding noise of varying strengths to the data.

By passing `--noise_from_expection true`, noise is added according to the forward noise process: if `--monte_carlo_samples null`, use the expectation of the noise process; otherwise, if `--monte_carlo_samples MONTE_CARLO_SAMPLES` and `MONTE_CARLO_SAMPLES` is an integer, estimate the noise process expectation using `MONTE_CARLO_SAMPLES` samples.

By passing `--noise_from_expection false`, obtain *"noised"* samples as intermediate solutions of the [probability flow ODE](https://arxiv.org/pdf/2011.13456). 

##### Integration-related arguments
Calculating IC with diffusion models corresponds to solving an ODE by numerical integration. As such, this involves trading off speed for accuracy.

Setting `--integration_params.n_runs N_RUNS` determines the number of Monte Carlo samples used for the [Skilling-Hutchingson divergence estimator](https://arxiv.org/pdf/1810.01367) in the [instantaneous change of variables formula](https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf). Setting this value low (e.g., setting it to `1`, which is feasible in some cases) speeds up performance.


Setting `--integration_params.solver euler` and `--integration_params.solver_kwargs "{'n_steps': 100}"`  uses a simple Euler solver with 100 steps (setting `n_steps` lower speeds up the estimation).

Setting `--integration_params.solver scipy` uses [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) as an integration backend. Additional arguments to `solve_ivp` (see documentation), can be passed using `--integration_params.solver_kwargs`.

##### Memory-related arguments
`--vmap_chunk_size VMAP_CHUNK_SIZE` determines the chunk size used when calculating IC for multiple time-steps in parallel. Set lower if you run into out-of-memory issues

`--bz BZ` Batch size for processing audio files. Setting this >1 can speed up computation if audio files are short and approximately uniform in length.

## Citation
If you use this project in your research, please cite the following paper:

```bibtex
@inproceedings{
   bjare2025perceptually,
   title={Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders},
   author={Mathias Rose Bjare and Giorgia Cantisani and Marco Pasini and Stefan Lattner and Gerhard Widmer},
   booktitle={NeurIPS - AI for Music Workshop},
   year={2025},
   url={https://openreview.net/forum?id=rXUKO0ysUy}
}
```

## License
This project is licensed under the CC BY-NC 4.0 License.

To obtain a commercial license, please contact [music@csl.sony.fr](mailto:music@csl.sony.fr).

