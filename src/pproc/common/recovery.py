import abc
import hashlib
import logging
import os
from typing import List, Optional

import yaml
from filelock import FileLock

from pproc.config.base import Recovery as BaseConfig

logger = logging.getLogger(__name__)


class BaseRecovery(abc.ABC):
    @abc.abstractmethod
    def existing_checkpoint(self, **checkpoint_identifiers) -> bool:
        pass

    @abc.abstractmethod
    def computed(self, **match) -> List[dict]:
        pass

    @classmethod
    def checkpoint_key(cls, checkpoint_identifiers: dict) -> str:
        return "/".join([f"{k}={v}" for k, v in checkpoint_identifiers.items()])

    @classmethod
    def checkpoint_identifiers(cls, key: str) -> dict:
        ret = {}
        for pair in key.split("/"):
            k, v = pair.split("=")
            ret[k] = v
        return ret

    @abc.abstractmethod
    def add_checkpoint(self, **checkpoint_identifiers):
        pass

    @abc.abstractmethod
    def clean_file(self):
        pass


class NullRecovery(BaseRecovery):
    def existing_checkpoint(self, **checkpoint_identifiers) -> bool:
        return False

    def computed(self, **match) -> List[dict]:
        return {}

    def add_checkpoint(self, **checkpoint_identifiers):
        pass

    def clean_file(self):
        pass


class Recovery(BaseRecovery):
    def __init__(self, root_dir: str, config: dict, recover: bool):
        """
        Class for writing out checkpoints and recovering computation from checkpoint file. The date and
        contents of the config are assumed to specify the run uniquely inside of root_dir.

        :param root_dir: directory to write checkpoint file to
        :param config: configuration dict for the run
        :param date: date and time of run
        :param recover: boolean specifying whether to retrieve checkpoints from file. Otherwise, existing
        checkpoints in the recovery file are deleted.
        """
        os.makedirs(root_dir, exist_ok=True)
        sha256 = hashlib.sha256(f"{yaml.dump(config)}".encode())
        self.filename = os.path.join(root_dir, f"{sha256.hexdigest()}.txt")
        self.checkpoints = []
        logger.info(
            f"Recovery: checkpoint file {self.filename}. Start from checkpoints: {recover}"
        )
        if recover:
            # Load from file if one exists
            if os.path.exists(self.filename):
                with open(self.filename, "rt") as f:
                    past_checkpoints = f.readlines()
                self.checkpoints += [x.rstrip("\n") for x in past_checkpoints]

        else:
            self.clean_file()
        self.lock = FileLock(self.filename + ".lock", thread_local=False)

    def computed(self, **match) -> List[dict]:
        ret = []
        for x in self.checkpoints:
            x_id = self.checkpoint_identifiers(x)
            if len(match) == 0 or (all(x_id[k] == str(v) for k, v in match.items())):
                ret.append(x_id)
        return ret

    def existing_checkpoint(self, **checkpoint_identifiers) -> bool:
        """
        Returns whether a checkpoint for the checkpoint_identifiers exists

        :param checkpoint_identifiers: unique list of parameters specifying
        checkpoint
        :return: bool for existence of checkpoint
        """
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        return checkpoint in self.checkpoints

    def add_checkpoint(self, **checkpoint_identifiers):
        """
        Add checkpoint to recover file. If it is an existing checkpoint
        then return

        :param checkpoint_identifiers: unique list of parameters specifying
        checkpoint
        """
        if self.existing_checkpoint(**checkpoint_identifiers):
            return
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        # Append new completed step to file

        with self.lock:
            logger.info(f"Adding checkpoint {checkpoint}")
            with open(self.filename, "at") as f:
                f.write(checkpoint + "\n")
            self.checkpoints.append(checkpoint)

    def clean_file(self):
        """
        Deletes existing recovery file if it exists
        """
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.filename + ".lock"):
            os.remove(self.filename + ".lock")


def create_recovery(config: BaseConfig) -> BaseRecovery:
    if config.recovery.enable_checkpointing:
        root_dir = config.recovery.root_dir or os.getcwd()
        return Recovery(
            root_dir,
            config.model_dump(exclude_defaults=True),
            config.recovery.from_checkpoint,
        )
    return NullRecovery()
