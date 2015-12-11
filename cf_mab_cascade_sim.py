from datetime import datetime
import cPickle as pickle
from cf_rec import CollaborativeFilter
from recommender_mab import RecommenderMAB
import load_categories as lc

tk_file = open('data/tk_test.pkl', 'rb')
tk = pickle.load(tk_file)

# Load categories hierarchy
print 'Loading Categories Hierarchy', datetime.now()
h = lc.load()

n_trials = tk.shape[0]
returned_recs = [None for i in range(n_trials)]
n_recs = [0.0 for i in range(n_trials)]
recovered_venues = [0.0 for i in range(n_trials)]
precision = 0

cf = CollaborativeFilter('train')
rm = RecommenderMAB(h.keys())

progress = tk.shape[0]
for cnt, (r_ind, r) in enumerate(tk.iterrows()):
    if cnt % (progress/10) == 0:
        print 'Progress:', cnt*100/progress, '%', datetime.now()
    
    u = r['user_id']
    recs = cf.recommend(u, None)
    recs = rm.select_arms(recs, 10)
    
    returned_recs[cnt] = recs
    n_recs[cnt] = len(recs)
    recs = set([t[0] for t in recs])
    
    # Reward
    venue = r['venue_id']
    if venue in recs:
        recovered = 1
    else:
        recovered = 0
    
    recovered_venues[cnt] = recovered
    
precision = sum(recovered_venues)/float(sum(n_recs))
recall = sum(recovered_venues)/float(n_trials)
print precision, recall
