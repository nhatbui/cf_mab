from datetime import datetime
import numpy as np
import categories_hierarchy as ch
from container_mab import ContainerMAB


def normalize(values):
    v = np.linalg.norm([v[1] for v in values])
    return [(val[0], val[1]/v) for val in values]


def train(events):
    # Load Categories Hierarchy
    print 'Loading Categories Hierarchy', datetime.now()
    h = ch.load()
    categories = h.keys()
    mab = ContainerMAB(n_arms=len(categories), n_bins=24)

    n_trials = events.shape[0]
    returned_cate = [None for i in range(n_trials)]
    rewards = [0.0 for i in range(n_trials)]
    cumulative_rewards = [0.0 for i in range(n_trials)]

    progress = n_trials
    for cnt, (r_ind, r) in enumerate(events.iterrows()):
        if cnt % (progress/10) == 0:
            print '    Progress:', cnt*100/progress, '%', datetime.now()

        t = r['datetime']
        cate_idx = mab.select_arms(int(t.hour))
        category = categories[cate_idx]

        returned_cate[cnt] = cate_idx

        # Compute Reward
        v_category = r['venue_category_id']
        if v_category in h:
            reward = h.dist_to_LCA(category, v_category, 0)
            reward = 1 - (reward/3)**2
        else:
            reward = 0
        mab.update(t.hour, cate_idx, reward)

        # Compute Cumulative Reward
        rewards[cnt] = reward
        if cnt == 0:
            cumulative_rewards[cnt] = reward
        else:
            cumulative_rewards[cnt] = cumulative_rewards[cnt-1] + reward
    stats = {
        'n_trials': n_trials,
        'pulled_arms': returned_cate,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards
    }
    return mab, stats
