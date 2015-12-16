from datetime import datetime
import pandas as pd
from learners.cf import CollaborativeFilter, NoPreferencesError

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
cf = CollaborativeFilter(tk_train)

# Statistics
n_trials = tk_test.shape[0]
returned_recs = [None for i in range(n_trials)]
n_recs = [0.0 for i in range(n_trials)]
recovered_venues = [0.0 for i in range(n_trials)]

print 'Beginning Test'
progress = n_trials
for cnt, (date_time, event) in enumerate(tk_test.iterrows()):
    if cnt % (progress/10) == 0:
        print 'Progress:', cnt*100/progress, '%', datetime.now()
    # Get set of Recommendations
    u = event['user_id']
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
    venue = event['venue_id']
    if venue in recs:
        recovered_venues[cnt] = 1
    else:
        recovered_venues[cnt] = 0

precision = sum(recovered_venues)/float(sum(n_recs))
recall = sum(recovered_venues)/float(n_trials)
print precision, recall
