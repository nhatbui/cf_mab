import os.path
from datetime import datetime
import cPickle as pickle
from cf_rec import CollaborativeFilter
from contextual_mab import ContextualMAB
import load_categories as lc
import train_contextmab as train


tk_file = open('data/tk_test.pkl', 'rb')
tk = pickle.load(tk_file)

# Load categories hierarchy
print 'Loading Categories Hierarchy', datetime.now()
h = lc.load()
categories = h.keys()

# Load Venue to Category Mapping
print 'Loading Venue Categories', datetime.now()
venues_to_categories_map_file = open('data/venue_to_categories_map.pkl', 'rb')
venues_to_categories_map = pickle.load(venues_to_categories_map_file)

# Statistics
n_trials = tk.shape[0]
returned_recs = [None for i in range(n_trials)]
n_recs = [0.0 for i in range(n_trials)]
recovered_venues = [0.0 for i in range(n_trials)]
precision = 0

cf = CollaborativeFilter('train')
cmab = ContextualMAB(len(h))
if not os.path.isfile('data/trained_cmab.pkl'):
    print 'Training Bandit...'
    cmab = train.train_cmab(cmab, 'data/tk_train.pkl')
    with open('data/trained_cmab.pkl', 'wb') as f:
        pickle.dump(cmab, f)
else:
    with open('data/trained_cmab.pkl', 'rb') as f:
        cmab = pickle.load(f)


progress = tk.shape[0]
for cnt, (r_ind, r) in enumerate(tk.iterrows()):
    if cnt % (progress/10) == 0:
        print '    Progress:', cnt*100/progress, '%', datetime.now()
    
    # Get set of recommendations
    u = r['user_id']
    recs = cf.recommend(u, None)
    
    # Get Context Rating
    t = r['datetime']
    cate_idx = cmab.select_arms(int(t.hour))
    rec_category = categories[cate_idx]
    
    # Re-calc Rec-Values with Context Rating
    new_recs = []
    for rec in recs:
        venue_id = rec[0]
        score = rec[1]
        if venue_id in venues_to_categories_map:
            category = venues_to_categories_map[venue_id][-1]
            if category in h:
                level_dist = h.dist_to_LCA(rec_category, category, 0)
                multiplier = 1 - (level_dist/3)**2 # [0,1]
                score *= multiplier
        new_recs.append((venue_id, score))
    new_recs = sorted(new_recs, key=lambda x: x[1], reverse=True)
    recs = new_recs[:20]
    
    # Record our recommendations
    returned_recs[cnt] = recs
    n_recs[cnt] = len(recs)
    
    # Check if we guessed correctly
    recs = set([rec[0] for rec in recs])
    venue = r['venue_id']
    if venue in recs:
        recovered_venues[cnt] = 1
    else:
        recovered_venues[cnt] = 0
    
        
    # Compute Reward
    v_category = r['venue_category_id']
    if v_category in h:
        reward = h.dist_to_LCA(rec_category, v_category, 0)
        reward = 1 - (reward/3)**2
    else:
        reward = 0
    cmab.update(t.hour, cate_idx, reward)
    
precision = sum(recovered_venues)/float(sum(n_recs))
recall = sum(recovered_venues)/float(n_trials)
print precision, recall

import matplotlib.pyplot as plt
v = {hr: cmab.values[hr] for hr in range(24)}
v_df = pd.DataFrame(v)
plt.figure(0)
for i in range(v_df.shape[0]):
    plt.plot(range(24), v_df.loc[i])
c = {hr: cmab.counts[hr] for hr in range(24)}
c_df = pd.DataFrame(c)
plt.figure(0)
for i in range(c_df.shape[0]):
    plt.plot(range(24), c_df.loc[i])
