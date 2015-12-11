import math
import numpy as np


def normalize(values):
    v = np.linalg.norm([v[1] for v in values])
    return [(val[0], val[1]/v) for val in values]

class ContextualMAB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.values = [[0.0]*self.n_arms for hour in range(24)]
        self.counts = [[1.0]*self.n_arms for hour in range(24)]
                     #OR [0.0 for arm in range(self.n_arms)]
        self.total_count = 0
    
    def select_arms(self, hour):
        self.total_count += 1
        max_val = 0
        best_arm = 0
        #print self.total_count, hour
        #print self.values
        #print self.counts
        for arm in range(self.n_arms):
            #print arm
            val = self.values[hour][arm] + math.sqrt(2*math.log(self.total_count)/self.counts[hour][arm])
            if val > max_val:
                max_val = val
                best_arm = arm
        self.counts[hour][best_arm] += 1
        return best_arm
        
    def update(self, hour, chosen_arm, reward):
        n = self.counts[hour][chosen_arm]
        prev_val = self.values[hour][chosen_arm]
        self.values[hour][chosen_arm] = (n - 1)/float(n)*prev_val + (1/float(n))*reward
