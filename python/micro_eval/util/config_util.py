import json
import os
import typing

from . import get_repo_root


class Config(dict):
  """Extends dict to add path lookup capabilities."""

  @classmethod
  def load(cls, path : str):
    """Load JSON dict from path.

    Params
    ------
    path : str
        The path to the JSON file. The JSON file should contain a dict at the
        top-level.

    Returns
    -------
    Config :
        The loaded Config instance.
    """
    with open(path) as f:
      obj = json.load(f)

    if not isinstance(obj, dict):
      raise ValueError(f'Expected JSON file to contain a dict, got {obj!r}')

    return cls(path, os.path.dirname(path), obj)

  def __init__(self, config_file_path, base_path, data):
    super(Config, self).__init__(data)
    self._config_file_path = config_file_path
    self._base_path = base_path

    if config_file_path is not None and base_path is not None:
      assert os.path.realpath(os.path.dirname(config_file_path)) == os.path.realpath(base_path)

  def relpath(self, key : typing.Union[str, int, bool]) -> str:
    return os.path.join(self._base_path, self[key])

  @property
  def config_file_path(self):
    return self._config_file_path
