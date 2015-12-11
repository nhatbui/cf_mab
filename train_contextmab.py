import cPickle as pickle
import load_categories as lc
from datetime import datetime


# training_set_filepath'data/tk_train.pkl'
def train_cmab(cmab, training_set_filepath):
    # Loading Training Data
    tk_file = open(training_set_filepath, 'rb')
    tk = pickle.load(tk_file)
    
    # Load Categories Hierarchy
    print 'Loading Categories Hierarchy', datetime.now()
    h = lc.load()
    categories = h.keys()

    # Statistics
    n_trials = tk.shape[0]
    returned_cate = [None for i in range(n_trials)]
    rewards = [0.0 for i in range(n_trials)]
    cumulative_rewards = [0.0 for i in range(n_trials)]
    
    print 'Beginning Training'
    progress = tk.shape[0]
    for cnt, (r_ind, r) in enumerate(tk.iterrows()):
        if cnt % (progress/10) == 0:
            print '    Progress:', cnt*100/progress, '%', datetime.now()
    
        t = r['datetime']
        cate_idx = cmab.select_arms(int(t.hour))
        category = categories[cate_idx]
    
        returned_cate[cnt] = cate_idx
    
        # Compute Reward
        v_category = r['venue_category_id']
        if v_category in h:
            reward = h.dist_to_LCA(category, v_category, 0)
            reward = 1 - (reward/3)**2
        else:
            reward = 0
        cmab.update(t.hour, cate_idx, reward)
    
        # Compute Cumulative Reward
        rewards[cnt] = reward
        if cnt == 0:
            cumulative_rewards[cnt] = reward
        else:
            cumulative_rewards[cnt] = cumulative_rewards[cnt-1] + reward
    return cmab

    #import matplotlib.pyplot as plt
    #plt.figure(0)
    #plt.plot(range(n_trials), cumulative_rewards)
    #plt.show()
