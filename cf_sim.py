from datetime import datetime
import cPickle as pickle
from cf_rec import CollaborativeFilter, NoPreferencesError

tk_file = open('data/tk_test.pkl', 'rb')
tk = pickle.load(tk_file)


cf = CollaborativeFilter('train')

# Statistics
n_trials = tk.shape[0]
returned_recs = [None for i in range(n_trials)]
n_recs = [0.0 for i in range(n_trials)]
recovered_venues = [0.0 for i in range(n_trials)]

progress = tk.shape[0]
for cnt, (r_ind, r) in enumerate(tk.iterrows()):
    if cnt % (progress/10) == 0:
        print 'Progress:', cnt*100/progress, '%', datetime.now()
    # Get set of Recommendations
    u = r['user_id']
    try:
        recs = cf.recommend(u, 20)
    except NoPreferencesError as e:
        print 'user', e.value, 'has no preferences.'
        continue
    
    # Record recommendations
    returned_recs[cnt] = recs
    n_recs[cnt] = len(recs)
    
    # Compute Reward
    recs = set([t[0] for t in recs])
    venue = r['venue_id']
    if venue in recs:
        recovered_venues[cnt] = 1
    else:
        recovered_venues[cnt] = 0
    
precision = sum(recovered_venues)/float(sum(n_recs))
recall = sum(recovered_venues)/float(n_trials)
print precision, recall
