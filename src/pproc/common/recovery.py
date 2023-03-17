import os
import datetime 
import hashlib
from typing import List

class Recovery:
    def __init__(self, root_dir: str, config_file: str, date: datetime.datetime, recover: bool = True):
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        sha256 = hashlib.sha256()
        with open(config_file, 'rb') as f:
            sha256.update(f.read())
        self.filename = os.path.join(root_dir, f"{date.strftime('%Y%m%d%H')}{sha256.hexdigest()}.txt") 
        self.checkpoints = []
        if recover:
            print(f"Recovery: checkpoints from file {self.filename}")
            # Load from file if one exists
            if os.path.exists(self.filename):
                with open(self.filename, 'rt') as f:
                    past_checkpoints = f.readlines()
                self.checkpoints += [x.rstrip('\n') for x in past_checkpoints]
        else:
            print(f"Recovery: cleaning checkpoints in file {self.filename}")
            self.clean()

    def existing_checkpoint(self, *checkpoint_identifiers) -> bool:
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        return checkpoint in self.checkpoints

    def last_checkpoint(self):
        if len(self.checkpoints) == 0:
            return None
        return self.checkpoints[-1]

    @classmethod
    def checkpoint_key(cls, checkpoint_identifiers: List):
        return '/'.join(map(str, checkpoint_identifiers))
    
    @classmethod
    def checkpoint_identifiers(cls, key: str) -> List[str]:
        return key.split("/")
    
    def add_checkpoint(self, *checkpoint_identifiers):
        checkpoint = self.checkpoint_key(checkpoint_identifiers)
        # Append new completed step to file
        with open(self.filename, 'at') as f:
            f.write(checkpoint + '\n')
        self.checkpoints.append(checkpoint)


    def clean(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)