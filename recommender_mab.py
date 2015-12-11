import math
from datetime import datetime
import cPickle as pickle
import numpy as np


def normalize(values):
    v = np.linalg.norm([v[1] for v in values])
    return [(val[0], val[1]/v) for val in values]

class RecommenderMAB:
    def __init__(self, arm_keys):
        #self.n_arms = n_arms
        self.counts = {}
        for k in arm_keys:
            self.counts[k] = 0
        #self.values = [0.0 for col in range(self.n_arms)]
        #self.set_baseline = False
        
        # Load Venue to Category Mapping
        print 'Loading Venue Categories', datetime.now()
        venues_file = open('data/venues.pkl', 'rb')
        self.venues = pickle.load(venues_file)
        self.total_count = 0
    
    def select_arms(self, arms, n_arms=10):
        self.total_count += 1
        #normalize values
        arms = normalize(arms)
        new_values = []
        for arm in arms:
            venue_id = arm[0]
            value = arm[1]
            if venue_id not in self.venues:
                new_values.append((venue_id, value))
            else:
                category = self.venues[venue_id]
                cate_count = self.counts[category]
                new_value = value + math.sqrt(2*math.log(self.total_count)/
                    float(cate_count))
                new_values.append((venue_id, new_value))
        
        new_values = sorted(new_values, key= lambda x: x[1])
        return new_values[:n_arms]
