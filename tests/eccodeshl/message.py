
import eccodes

class Message:
    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        eccodes.codes_release(self.handle)

    def get_array(self, name):
        return eccodes.codes_get_array(self.handle, name)
