import os
import datetime
import hashlib
from typing import List
from filelock import FileLock


class Recovery:
    def __init__(
        self, root_dir: str, config_file: str, date: datetime.datetime, recover: bool
    ):
        """
        Class for writing out checkpoints and recovering computation from checkpoint file. The date and
        contents of the config_file are assumed to specify the run uniquely inside of root_dir.

        :param root_dir: directory to write checkpoint file to
        :param config_file: file used to set up run
        :param date: date and time of run
        :param recover: boolean specifying whether to retrieve checkpoints from file. Otherwise, existing
        checkpoints in the recovery file are deleted.
        """
        os.makedirs(root_dir, exist_ok=True)
        sha256 = hashlib.sha256()
        with open(config_file, "rb") as f:
            sha256.update(f.read())
        self.filename = os.path.join(
            root_dir, f"{date.strftime('%Y%m%d%H')}{sha256.hexdigest()}.txt"
        )
        self.checkpoints = []
        print(
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
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = FileLock(self.filename + ".lock")
        return self._lock

    def existing_checkpoint(self, *checkpoint_identifiers) -> bool:
        """
        Returns whether a checkpoint for the checkpoint_identifiers exists

        :param checkpoint_identifiers: unique list of parameters specifying
        checkpoint
        :return: bool for existence of checkpoint
        """
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        return checkpoint in self.checkpoints

    def last_checkpoint(self) -> str:
        """
        Returns last checkpoint
        """
        if len(self.checkpoints) == 0:
            return None
        return self.checkpoints[-1]

    @classmethod
    def checkpoint_key(cls, checkpoint_identifiers: List):
        return "/".join(map(str, checkpoint_identifiers))

    @classmethod
    def checkpoint_identifiers(cls, key: str) -> List[str]:
        return key.split("/")

    def add_checkpoint(self, *checkpoint_identifiers):
        """
        Add checkpoint to recover file. If it is an existing checkpoint
        then return

        :param checkpoint_identifiers: unique list of parameters specifying
        checkpoint
        """
        if self.existing_checkpoint(*checkpoint_identifiers):
            return
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        # Append new completed step to file

        with self.lock:
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
