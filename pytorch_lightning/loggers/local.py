##########################
# constants
import pickle
from argparse import Namespace
import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np

import pandas as pd

import os

import shutil
from natsort import natsorted
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import _logger as log


# -----------------------------
# Experiment object
# -----------------------------
class Experiment(object):

    def __init__(self, save_dir=None, name='default', debug=False, version=None, description=None, autosave=False):
        """
        A new Experiment object defaults to 'default' unless a specific name is provided
        If a known name is already provided, then the file version is changed
        :param name:
        :param debug:
        """

        # change where the save dir is if requested

        if save_dir is not None:
            global _ROOT
            _ROOT = save_dir

        self.save_dir = Path(save_dir)
        self.no_save_dir = save_dir is None
        self.metrics = []
        self.name = name
        self.tags = {}
        self.debug = debug
        self.version = version
        self.description = description
        self.exp_hash = '{}_v{}'.format(self.name, version)
        self.created_at = str(datetime.utcnow())
        self.process = os.getpid()
        self.autosave = autosave

        # when debugging don't do anything else
        if debug:
            return

        # update version hash if we need to increase version on our own
        # we will increase the previous version, so do it now so the hash
        # is accurate
        if version is None:
            old_version = self.__get_last_experiment_version()
            self.exp_hash = '{}_v{}'.format(self.name, old_version + 1)
            self.version = old_version + 1

        # create a new log dir
        self.data_path.mkdir(parents=True, exist_ok=True)

        # when we have a version, load it
        if self.version is not None:

            # when no version and no file, create it
            if not (self.data_path / 'meta.experiment.csv').exists():
                self.__create_exp_file(self.version)
            else:
                try:
                    # otherwise load it
                    self.__load()
                except json.decoder.JSONDecodeError:
                    # File was empty, create it from scratch
                    (self.data_path / 'meta.experiment.csv').unlink()
                    self.__create_exp_file(self.version)
        else:
            # if no version given, increase the version to a new exp
            # create the file if not exists
            old_version = self.__get_last_experiment_version()
            self.version = old_version
            self.__create_exp_file(self.version + 1)

    def on_exit(self):
        pass

    # --------------------------------
    # FILE IO UTILS
    # --------------------------------
    def __create_exp_file(self, version):
        """
        Recreates the old file with this exp and version
        :param version:
        :return:
        """

        try:
            # if no exp, then make it
            path = self.data_path / 'meta.experiment.csv'
            path.touch(exist_ok=True)

            self.version = version

            # make the directory for the experiment media assets name
            self.media_path.mkdir(parents=True, exist_ok=True)

        except FileExistsError:
            # file already exists (likely written by another exp. In this case disable the experiment
            self.debug = True

    def __get_last_experiment_version(self):

        exp_cache_file = self.data_path.parent
        last_version = -1

        version = natsorted([x.name for x in exp_cache_file.iterdir() if 'version_' in x.name])[-1]
        last_version = max(last_version, int(version.split('_')[1]))

        return last_version

    def log_metrics(self, metrics_dict):
        """
        Adds a json dict of metrics.

        >> e.log({"loss": 23, "coeff_a": 0.2})

        :param metrics_dict:

        :return:
        """
        if self.debug:
            return

        # timestamp
        if 'created_at' not in metrics_dict:
            metrics_dict['created_at'] = str(datetime.utcnow())

        self.__convert_numpy_types(metrics_dict)

        self.metrics.append(metrics_dict)

        if self.autosave:
            self.save()

    @staticmethod
    def __convert_numpy_types(metrics_dict):
        for k, v in metrics_dict.items():
            if v.__class__.__name__ == 'float32':
                metrics_dict[k] = float(v)

            if v.__class__.__name__ == 'float64':
                metrics_dict[k] = float(v)

    def save(self):
        """
        Saves current experiment progress
        :return:
        """
        if self.debug:
            return

        metrics_file_path = self.data_path / 'metrics.csv'
        meta_tags_path = self.data_path / 'meta_tags.csv'
        meta_exp_path = self.data_path / 'meta.experiment.csv'
        pickle_self_path = self.data_path / 'exp.pikle'


        # save the experiment meta file
        obj = {
            'name': self.name,
            'version': self.version,
            'tags_path': str(meta_tags_path),
            'metrics_path': str(metrics_file_path),
            'description': self.description,
            'created_at': self.created_at,
            'exp_hash': self.exp_hash
        }
        with meta_exp_path.open('w') as f:
            json.dump(obj, f, ensure_ascii=False)

        # save the metatags file
        df = pd.DataFrame({'key': list(self.tags.keys()), 'value': list(self.tags.values())})
        with meta_tags_path.open('w') as f:
            df.to_csv(f, index=False)

        # save the metrics data
        df = pd.DataFrame(self.metrics)
        with Path(f'{metrics_file_path}_tmp').open('a') as f:
            df.to_csv(f, index=False)
        shutil.move(f'{metrics_file_path}_tmp', str(metrics_file_path))

        # pickle self
        with pickle_self_path.open('wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __load(self):
        # load .experiment file
        with (self.data_path / 'meta.experiment.csv').open('r') as f:
            data = json.load(f)
            self.name = data['name']
            self.version = data['version']
            self.created_at = data['created_at']
            self.description = data['description']
            self.exp_hash = data['exp_hash']

        # load .tags file
        meta_tags_path = self.data_path / 'meta_tags.csv'
        df = pd.read_csv(meta_tags_path)
        self.tags_list = df.to_dict(orient='records')
        self.tags = {}
        for d in self.tags_list:
            k, v = d['key'], d['value']
            self.tags[k] = v

        # load metrics
        metrics_file_path = self.data_path / 'metrics.csv'
        try:
            df = pd.read_csv(metrics_file_path)
            self.metrics = df.to_dict(orient='records')

            # remove nans and infs
            for metric in self.metrics:
                to_delete = []
                for k, v in metric.items():
                    if np.isnan(v) or np.isinf(v):
                        to_delete.append(k)
                for k in to_delete:
                    del metric[k]

        except Exception:
            # metrics was empty...
            self.metrics = []

    @property
    def data_path(self) -> Path:
        if self.no_save_dir:
            return _ROOT / 'local_experiment_data' / self.name / f'version_{self.version}'
        else:
            return _ROOT / self.name / f'version_{self.version}'

    @property
    def media_path(self):
        return self.data_path / 'media'

    # ----------------------------
    # OVERWRITES
    # ----------------------------
    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

    def __hash__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)


##########################
class LocalLogger(LightningLoggerBase):

    @property
    def name(self) -> str:
        return self._name

    @property
    def logdir(self) -> Path:
        return self.experiment.data_path

    @property
    def experiment(self) -> Experiment:
        r"""

        Actual TestTube object. To use TestTube features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_test_tube_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            autosave=False
        )
        return self._experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step:
            metrics.update(step=step)
        self.experiment.log_metrics(metrics)

    def finalize(self, status: str) -> None:
        self.experiment.tags.update(exp_status=status)
        self.experiment.save()

    def log_image(self, name, image, step=None, **kwargs):
        name = f'{step}_{name}' if step is not None else name
        image.savefig(str(self.experiment.media_path / name), **kwargs)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            self.experiment.tags.update(key=val)

    @property
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = self.save_dir / self.name

        if not root_dir.is_dir():
            log.warning(f'Missing logger folder: {root_dir}')
            return 0

        existing_versions = []
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith("version_"):
                existing_versions.append(int(d.name.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def __init__(self, save_dir: str, name: str = "default", description: Optional[str] = None,
                 debug: bool = False, version: Optional[int] = None, **kwargs):
        super(LocalLogger, self).__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self._name = name
        self.description = description
        self.debug = debug
        self._version = version
        self._experiment = None
