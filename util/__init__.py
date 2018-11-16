class LinearDecay(object):
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.curr_steps = 0
        self.curr_epsilon = start

    def __call__(self):
        self.curr_steps += 1
        interp = min(self.curr_steps, self.steps) / self.steps
        self.curr_epsilon = (1 - interp) * self.start + interp * self.end

        return self.curr_epsilon
