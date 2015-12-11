from datetime import datetime
import pandas as pd
import numpy as np
import math
import cPickle as pickle


class CFOffline:
    def __init__(self, tk, tag=None):
        self.tag = tag
        self.tk = tk
    
    def run(self):
        # Useful data
        categories = self.tk['venue_category_id'].unique()
        users = self.tk['user_id'].unique()
        
        # User-Category Matrices
        print 'Creating User-Category Matrices'
        tk_ucm = {}
        progress = len(categories)
        for idx, category in enumerate(categories):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now() 
            df = self.tk[self.tk['venue_category_id'] == category]
            tk_ucm[category] = pd.crosstab(df['user_id'], df['venue_id'])
        
        # HITS-based inference for 'local experts'
        print 'Performing Iterative Inference for Local Experts...'
        experts = {}
        interest = {}
        progress = len(categories)
        for idx, category in enumerate(categories):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now() 
            
            M = tk_ucm[category].as_matrix()
            MT = M.transpose()
            MTM = np.dot(MT, M)  # MT*M
            MMT = np.dot(M, MT)  # M*MT
            
            # Initialize authority and hub scores as the # of user's visits
            # Hub
            H = M.sum(axis=1)
            H /= np.linalg.norm(H)
            # Authority
            A = MT.sum(axis=1)
            A /= np.linalg.norm(A)
            
            # Power iteration method
            for _ in range(100):
                A = np.dot(MTM, A)
                # This could've been np.dot(MT, H)
                A /= np.linalg.norm(A)
        
                H = np.dot(MMT, H)
                # This could've been np.dot(M, A)
                H /= np.linalg.norm(H)
                
            experts[category] = pd.DataFrame(
                data=H,
                index=tk_ucm[category].index,
                columns=['hub']
            )
            experts[category].sort(['hub'], ascending=False)
            interest[category] = pd.DataFrame(
                data=A,
                index=tk_ucm[category].columns,
                columns=['authority']        
            )
        with open('data/experts_' + self.tag + '.pkl', 'wb') as f:
            pickle.dump(experts, f)
        
        # User Preference Weight
        print 'Calculating TF...'
        nUsers = len(users)  # number of users in system
        user_tf = {}  # term frequency
        user_visits = {}  # total number of visits by a user
        progress = nUsers
        for idx, user in enumerate(users):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now()
            df = self.tk[self.tk['user_id'] == user].groupby('venue_category_id').agg(
                {'venue_category_id': np.size}
            )
            user_visits[user] = df['venue_category_id'].sum()
            user_tf[user] = df
        
        print 'Calculating IDF...'
        cate_idf = {}  # inverse document frequency
        progress = len(categories)
        for idx, category in enumerate(categories):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now() 
            cate_idf[category] = tk_ucm[category].shape[0]
            
        print 'Calculating User\'s Preference Weight'
        user_c_weight = {} # user's preference weight
        progress = nUsers
        for idx, user in enumerate(users):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now()     
            
            user_c_weight[user] = {}
            for category in categories:
                if category in user_tf[user].index:
                    user_c_weight[user][category] = \
                        user_tf[user].loc[category, 'venue_category_id'] *\
                        math.log(float(nUsers)/cate_idf[category]) / \
                        user_visits[user]
        with open('data/ucw_' + self.tag + '.pkl', 'wb') as f:
            pickle.dump(user_c_weight, f)

if __name__ == '__main__':
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
    tk_path = './dataset_tsmc2014/dataset_TSMC2014_TKY.txt'
        
    print 'Loading data:', tk_path 
    tk = pd.read_csv(tk_path, names=header, header=0, delimiter='\t')
    tk['datetime'] = pd.to_datetime(tk['datetime'].values, dt_format)
    tk.index = tk['datetime']
    
    train = tk[:'2013-2-15']
    with open('data/tk_train.pkl', 'wb') as f:
        pickle.dump(train, f)
        
    test = tk['2013-2-15':]
    with open('data/tk_test.pkl', 'wb') as f:
        pickle.dump(test, f)
    
    ct = pd.crosstab(train['user_id'], train['venue_id'])
    with open('data/uv_count_train.pkl', 'wb') as f:
        pickle.dump(ct, f)
        
    venue_to_category = {}
    venue_groups = train.groupby('venue_id')
    for group in venue_groups.groups:
        venue_to_category[group] = venue_groups.get_group(group)['venue_category_id'].unique()
    with open('data/venue_to_categories_map.pkl', 'wb') as f:
        pickle.dump(venue_to_category, f)
        
    venues_in_category = {}
    category_groups = train.groupby('venue_category_id')
    for group in category_groups.groups:
        venues_in_category[group] = category_groups.get_group(group)['venue_id'].unique()
    with open('data/venues_in_category.pkl', 'wb') as f:
        pickle.dump(venues_in_category, f)

    cfo = CFOffline(train, 'train')
    cfo.run()