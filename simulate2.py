import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from learners.cf import CollaborativeFilter, NoPreferencesError
import learners.temporal_mab as time_bandit

header = [
    'user_id',             # User ID (anonymized)
    'venue_id',            # Venue ID (Foursquare)
    'venue_category_id',   # Venue category ID (Foursquare)
    'venue_category_name', # Venue category name (Fousquare)
    'lat',                 # Latitude
    'lon',                 # Longitude
    'timezone_offset',     # Timezone offset in minutes (The offset in minutes
                           # between when this check-in occurred and the same
                           # time in UTC)
    'datetime'             # UTC time
]
dt_format = '%a %b %d %H:%M:%S %z %Y'
usr_loc_hist_path = './data/dataset_tsmc2014/dataset_TSMC2014_TKY.txt'
usr_loc_hist = pd.read_csv(usr_loc_hist_path,
                           names=header,
                           header=0,
                           delimiter='\t')
usr_loc_hist['datetime'] = pd.to_datetime(
    usr_loc_hist['datetime'].values,
    dt_format
)
usr_loc_hist.index = usr_loc_hist['datetime']
tk_train = usr_loc_hist[:'2013-02-14']
tk_test = usr_loc_hist['2013-02-14':]
#tk_train = usr_loc_hist[:'2012-04-04']
#tk_test = usr_loc_hist['2012-04-04':'2012-04-05']

cmab, stats = time_bandit.train(tk_train)
cf = CollaborativeFilter(tk_train)

# Load categories hierarchy
h = cf.ch
categories = h.keys()

# Load Venue to Category Mapping
venues_to_categories_map = usr_loc_hist['venue_category_id']
venues_to_categories_map.index = usr_loc_hist['venue_id']
venues_to_categories_map = venues_to_categories_map.drop_duplicates()

# Statistics
n_trials = tk_test.shape[0]
returned_recs = [None for i in range(n_trials)]
n_recs = [0.0 for i in range(n_trials)]
recovered_venues = [0.0 for i in range(n_trials)]

print 'Beginning Test'
progress = tk_test.shape[0]
for cnt, (r_ind, r) in enumerate(tk_test.iterrows()):
    if cnt % (progress/10) == 0:
        print '    Progress:', cnt*100/progress, '%', datetime.now()
    
    # Get set of recommendations
    u = r['user_id']
    try:
        recs = cf.recommend(u, 20)
    except NoPreferencesError as e:
        print 'user', e.value, 'has no preferences.'
        continue
    
    # Get Context Rating
    t = r['datetime']
    cate_idx = cmab.select_arms(int(t.hour))
    rec_category = categories[cate_idx]
    
    # Re-calc Rec-Values with Context Rating
    new_recs = []
    for rec in recs:
        venue_id = rec[0]
        score = rec[1]
        if venue_id in venues_to_categories_map.index:
            category = venues_to_categories_map.loc[venue_id]
            if category in categories:
                level_dist = h.dist_to_LCA(rec_category, category, 0)
                multiplier = 1 - (level_dist/3)**2  # [0,1]
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
plt.show()
