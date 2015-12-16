import math


class ContainerMAB:
    def __init__(self, n_arms, n_bins):
        self.n_arms = n_arms
        self.values = [[0.0]*self.n_arms for b in range(n_bins)]
        self.counts = [[1.0]*self.n_arms for b in range(n_bins)]
        self.total_count = 0
    
    def select_arms(self, bin):
        self.total_count += 1
        max_val = 0
        best_arm = 0

        for arm in range(self.n_arms):
            val = self.values[bin][arm] + \
                  math.sqrt(
                      2 * math.log(self.total_count) / self.counts[bin][arm]
                  )
            if val > max_val:
                max_val = val
                best_arm = arm
        self.counts[bin][best_arm] += 1
        return best_arm
        
    def update(self, bin, chosen_arm, reward):
        n = self.counts[bin][chosen_arm]
        prev_val = self.values[bin][chosen_arm]
        self.values[bin][chosen_arm] = (n - 1) / float(n) * prev_val + \
                                       (1 / float(n)) * reward
