
class Window:
    def __init__(self, window_options):
        self.start = int(window_options['steps'][0])
        self.end = int(window_options['steps'][1])
        self.step = int(window_options['steps'][2])
        self.steps = list(range(self.start, self.end+self.step, self.step))
        window_size = self.end-self.start
        self.suffix = f"{window_size:0>3}_{self.start:0>3}h_{self.end:0>3}h"
        self.name = f"{self.start}-{self.end}"
