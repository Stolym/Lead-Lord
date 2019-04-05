#!/usr/bin/env python3

class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []
    
    def add_sample(self, sample):
        self.samples.append(sample)
        if (len(self.samples) > self.max_memory):
            self.samples.pop(0)

    def sample(self, no_samples):
        if (no_samples > len(self.samples)):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, no_samples)