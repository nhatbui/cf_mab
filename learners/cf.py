import numpy as np
import pandas as pd
import math
from datetime import datetime
import categories_hierarchy as hc

class NoPreferencesError(Exception):
    def __init__(self, user):
        self.value = user

    def __str__(self):
        return repr(self.value)


class CollaborativeFilter:
    def __init__(self, usr_loc_hist):
        # Load categories hierarchy
        self.ch = hc.load()
        # number of users in system
        self.n_users = len(usr_loc_hist['user_id'].unique())
        # user-venue count matrix
        self.uv_cnt = pd.crosstab(usr_loc_hist['user_id'],
                                  usr_loc_hist['venue_id'])
        # user location history
        self.usr_loc_hist = usr_loc_hist.groupby(['venue_category_id'])
        # venues in categories
        self.venues = {}
        # user's preference weight
        self.ucw = {}
        # category experts
        self.experts = {}
        # Train
        self.offline()

    def _HITS(self, A, MTM, H, MMT):
        # Power iteration method
        for _ in range(100):
            A = np.dot(MTM, A)
            # This could've been np.dot(MT, H)
            A /= np.linalg.norm(A)

            H = np.dot(MMT, H)
            # This could've been np.dot(M, A)
            H /= np.linalg.norm(H)
        return A, H

    def _TF_IDF(self, users, usr_venue_cnt):
        # User Preference Weight
        uw = {}
        for user in users:
            #pdb.set_trace()
            # times user went to a venue of category C
            uv_c = usr_venue_cnt.loc[user].sum()
            # total visits by user
            uv = self.uv_cnt.loc[user].sum()
            tf = float(uv_c)/uv

            # number of users who have visited a venue of category C
            visits_per_usr = usr_venue_cnt.sum(axis=1)
            total_cate_cnt = visits_per_usr[visits_per_usr > 0].count()
            idf = float(self.n_users)/total_cate_cnt

            uw[user] = tf*math.log(idf)
        return uw

    def offline(self):
        ucw = {}
        categories = self.usr_loc_hist.groups

        print 'Calculating Experts and User\'s Preference Weight'
        progress = len(categories)
        for idx, category in enumerate(categories):
            if idx % (progress/10) == 0:
                print '    Progress:', idx*100/progress, '%,', datetime.now()

            df = self.usr_loc_hist.get_group(category)
            #pdb.set_trace()
            category_venues = df['venue_id'].unique()
            self.venues[category] = category_venues
            ucm = self.uv_cnt[category_venues]

            # HITS-based inference for 'local experts'
            M = ucm.as_matrix()
            MT = M.transpose()
            MTM = np.dot(MT, M)  # MT*M
            MMT = np.dot(M, MT)  # M*MT

            # Initialize authority and hub scores as the # of user's visits
            # Hub
            hubs = M.sum(axis=1)
            hubs = np.divide(hubs, np.linalg.norm(hubs))
            # Authority
            auth = MT.sum(axis=1)
            auth = np.divide(auth, np.linalg.norm(auth))

            auth, hubs = self._HITS(auth, MTM, hubs, MMT)

            self.experts[category] = pd.DataFrame(
                data=hubs,
                index=ucm.index,
                columns=['hub']
            )
            self.experts[category].sort(['hub'], ascending=False, inplace=True)

            # get user weights for category
            users = df['user_id'].unique()
            ucw[category] = self._TF_IDF(users, ucm)
        self.ucw = pd.DataFrame.from_dict(ucw)

    def recommend(self, usr, n_recs=10, n_venues=10, debug=False):
        if usr not in self.uv_cnt.index:
            raise NoPreferencesError(usr)
        pref = [c for c in self.ucw.columns if self.ucw.loc[usr, c] > 0]
        todo = pref
    
        # For every user-preferred category,
        # return a set of candidate experts and venues for that category.
        if debug:
            print 'Finding Candidate Experts and Venues...'
        candidate_experts = set()
        candidate_venues_count = 0
        not_root = True
        while not_root:
            parent = None
            for u_cate in todo:
                if u_cate and u_cate in self.ch:
                    parent = self.ch[u_cate]['parent']
                elif u_cate:
                    # we shall skip categories unknown to our hierarchy
                    continue
    
                if not parent:
                    # made it to the top of the tree. Return
                    not_root = False
                    break
    
                # Get all categories in the same level
                children = self.ch[parent]['children']
                # find minumum weight for level by looking in user-category weights
                w_cate = [self.ucw.loc[usr, c] for c in children
                          if c in self.ucw.columns and \
                          not pd.isnull(self.ucw.loc[usr, c])]
                if len(w_cate) == 0:
                    continue
                w_min = min(w_cate)
    
                # Calculate an estimate of experts we should have for
                # this category.
                if not pd.isnull(self.ucw.loc[usr, u_cate]):
                    # User weight for category
                    user_cate_wt = self.ucw.loc[usr, u_cate]
                    k = int(user_cate_wt/w_min)
                else:
                    k = 1
    
                top_experts = self.experts[u_cate].head(k).index
    
                for expert in top_experts:
                    # nonzero() always returns a tuple.
                    candidate_venues_count += len(self.uv_cnt.loc[expert].nonzero()[0])
                candidate_experts.update(top_experts)
            if candidate_venues_count > n_venues:
                break
            else:
                todo = [parent]
    
        # Location Rating Inference for Similarity
        similarity = {}
        u_entropy = {}
        u_count = self.uv_cnt.loc[usr].sum()  # Count of all the visits a user has made
    
        if debug:
            print 'Calculating Similarity Amongst Experts...'
        progress = float(len(candidate_experts))
        for prog, expert in enumerate(candidate_experts):
            if debug:
                if prog % (progress/10) == 0:
                    print '    Progress:', prog*100/progress, '%', datetime.now()
    
            # Level sum and Entropy
            level_sum = {}
            entropy = {}
            count = self.uv_cnt.loc[expert].sum()
    
            expert_pref = [c for c in self.ucw.keys()
                           if self.ucw.loc[expert, c] > 0]
            overlapped_cates = np.intersect1d(
                pref, expert_pref, assume_unique=True
            )
            for node in overlapped_cates:
                if node not in self.ch.keys():
                    continue
                # The level which a category belongs on is defined by their
                # parent
                parent = self.ch[node]['parent']

                expert_w = self.ucw.loc[expert, node]
                u_wc = self.ucw.loc[usr, node]
                if parent in level_sum:
                    level_sum[parent] += min(expert_w, u_wc)
                else:
                    level_sum[parent] = min(expert_w, u_wc)
    
                idxs = self.venues[node]
                c_count = self.uv_cnt.loc[expert][idxs].sum()
                u_c_count = self.uv_cnt.loc[usr][idxs].sum()
    
                p = float(c_count)/count
                tp = float(u_c_count)/u_count
                if p == 0:
                    p = .01
                if tp == 0:
                    tp = .01
    
                if parent in entropy:
                    entropy[parent] -= p * math.log(p)
                    u_entropy[parent] -= tp * math.log(tp)
                else:
                    entropy[parent] = -p * math.log(p)
                    u_entropy[parent] = -tp * math.log(tp)
            similarity[expert] = [0]  # make it an array for Pandas reasons
            for level in level_sum.keys():
                similarity[expert][0] += \
                    self.ch.get_level_num(level) * level_sum[level] / \
                    (1+abs(u_entropy[level] - entropy[level]))
    
        # Location Rating Calculation
        v = self.uv_cnt.loc[similarity.keys()]
        venue_ids = v.columns
        sim = pd.DataFrame.from_dict(similarity)
        r = np.dot(sim.values,v.values)
    
        # get top N recommendations from r
        if r.size == 0:
            return []
        rec = zip(venue_ids.values.tolist(), r[0].tolist())
        rec = sorted(rec, key=lambda x: x[1], reverse=True)
        top = rec[:n_recs]
        return top
