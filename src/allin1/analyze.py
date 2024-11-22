import torch

from typing import List, Union
from tqdm import tqdm
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .visualize import visualize as _visualize
from .sonify import sonify as _sonify
from .helpers import (
  run_inference,
  expand_paths,
  check_paths,
  rmdir_if_empty,
  save_results,
)
from .utils import mkpath, load_result
from .typings import AnalysisResult, PathLike


def analyze(
  paths: Union[PathLike, List[PathLike]],
  demucs_paths: List[PathLike],
  out_dir: PathLike = None,
  visualize: Union[bool, PathLike] = False,
  sonify: Union[bool, PathLike] = False,
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demix',
  spec_dir: PathLike = './spec',
  keep_byproducts: bool = False,
  overwrite: bool = False,
  multiprocess: bool = True,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  """
  Analyzes the provided audio files and returns the analysis results.

  Parameters
  ----------
  paths : Union[PathLike, List[PathLike]]
      List of paths or a single path to the audio files to be analyzed.
  out_dir : PathLike, optional
      Path to the directory where the analysis results will be saved. By default, the results will not be saved.
  visualize : Union[bool, PathLike], optional
      Whether to visualize the analysis results or not. If a path is provided, the visualizations will be saved in that
      directory. Default is False. If True, the visualizations will be saved in './viz'.
  sonify : Union[bool, PathLike], optional
      Whether to sonify the analysis results or not. If a path is provided, the sonifications will be saved in that
      directory. Default is False. If True, the sonifications will be saved in './sonif'.
  model : str, optional
      Name of the pre-trained model to be used for the analysis. Default is 'harmonix-all'. Please refer to the
      documentation for the available models.
  device : str, optional
      Device to be used for computation. Default is 'cuda' if available, otherwise 'cpu'.
  include_activations : bool, optional
      Whether to include activations in the analysis results or not.
  include_embeddings : bool, optional
      Whether to include embeddings in the analysis results or not.
  demix_dir : PathLike, optional
      Path to the directory where the source-separated audio will be saved. Default is './demix'.
  spec_dir : PathLike, optional
      Path to the directory where the spectrograms will be saved. Default is './spec'.
  keep_byproducts : bool, optional
      Whether to keep the source-separated audio and spectrograms or not. Default is False.
  overwrite : bool, optional
      Whether to overwrite the existing analysis results or not. Default is False.
  multiprocess : bool, optional
      Whether to use multiprocessing for spectrogram extraction, visualization, and sonification. Default is True.

  Returns
  -------
  Union[AnalysisResult, List[AnalysisResult]]
      Analysis results for the provided audio files.
  """
  results = []
  # Clean up the arguments.
  spec_dir = mkpath(spec_dir)

  # Analyze the tracks that are not analyzed yet.
  # Extract spectrograms for the tracks that are not analyzed yet.
  spec_paths = extract_spectrograms(demucs_paths, spec_dir, multiprocess)

  # Load the model.
  model = load_pretrained_model(model_name=model, device=device)

  with torch.no_grad():
    pbar = tqdm(zip(demucs_paths, spec_paths), total=len(demucs_paths))
    for path, spec_path in pbar:
      pbar.set_description(f'Analyzing {path.name}')

      result = run_inference(
        path=path,
        spec_path=spec_path,
        model=model,
        device=device,
        include_activations=include_activations,
        include_embeddings=include_embeddings,
      )

      results.append(result)

  for path in spec_paths:
    path.unlink(missing_ok=True)
  rmdir_if_empty(spec_dir)

  return results[0]
