
class Window:
    def __init__(self, window_options, include_init=True):
        self.start = int(window_options['range'][0])
        self.end = int(window_options['range'][1])
        self.step = int(window_options['range'][2])
        if include_init:
            self.steps = list(range(self.start, self.end+self.step, self.step))
        else:
            self.steps = list(range(self.start+self.step, self.end+self.step, self.step))
        window_size = self.end-self.start
        self.suffix = f"{window_size:0>3}_{self.start:0>3}h_{self.end:0>3}h"
        self.name = f"{self.start}-{self.end}"
