import numpy as np
import pandas as pd
import cPickle as pickle
import math
from datetime import datetime
import load_categories as lc


class NoPreferencesError(Exception):
    def __init__(self, user):
        self.value = user

    def __str__(self):
        return repr(self.value)


class CollaborativeFilter:
    def __init__(self, tag):
        # Load offline data
        # Load user x venue count matrix
        print 'Loading User X Venue Count Matrix', datetime.now()
        uv_cnt_file = open('data/uv_count_' + tag + '.pkl', 'rb')
        self.uv_cnt = pickle.load(uv_cnt_file)
        
        # Load Venue to Category Mapping
        print 'Loading Venue Categories', datetime.now()
        venues_file = open('data/venues_in_category.pkl', 'rb')
        self.venues = pickle.load(venues_file)
        
        # Load user-category weights
        print 'Loading User-Category Weights', datetime.now()
        ucw_file = open('data/ucw_' + tag + '.pkl', 'rb')
        self.ucw = pickle.load(ucw_file)
        
        # Load experts
        print 'Loading Category Experts', datetime.now()
        experts_file = open('data/experts_' + tag + '.pkl', 'rb')
        self.category_experts = pickle.load(experts_file)
        
        # Load categories hierarchy
        print 'Loading Categories Hierarchy', datetime.now()
        self.h = lc.load()

    def recommend(self, u, n_recs=10, n_venues=10, debug=False):
        if u not in self.ucw:
            raise NoPreferencesError(u)
        pref = [pref for pref in self.ucw[u] if self.ucw[u][pref] > 0]
        todo = pref
    
        # For every user-preferred category,
        # return a set of candidate experts and venues for that category.
        if debug:
            print 'Finding Candidate Experts and Venues...'
        candidate_experts = set()
        candidate_venues_count = 0
        not_root = True
        while not_root:
            #pdb.set_trace()
            parent = None
            for u_cate in todo:
                if u_cate and u_cate in self.h:
                    parent = self.h[u_cate]['parent']
                elif u_cate:
                    # we shall skip categories unknown to our hierarchy
                    continue
    
                if not parent:
                    # made it to the top of the tree. Return
                    not_root = False
                    break
    
                # Get all categories in the same level
                children = self.h[parent]['children']
                # find minumum weight for level by looking in user-category weights
                w_cate = [self.ucw[u][cate] for cate in children if cate in self.ucw[u]]
                if len(w_cate) == 0:
                    continue
                w_min = min(w_cate)
    
                # Calculate an estimate of experts we should have for
                # this category.
                if u_cate in self.ucw[u]:
                    # User weight for category
                    user_cate_wt = self.ucw[u][u_cate]
                    k = int(user_cate_wt/w_min)
                else:
                    k = 1
    
                top_experts = self.category_experts[u_cate].head(k).index
    
                for expert in top_experts:
                    # nonzero() always returns a tuple.
                    candidate_venues_count += len(self.uv_cnt.loc[expert].nonzero()[0])
                candidate_experts.update(top_experts)
            if candidate_venues_count > n_venues:
                break
            else:
                todo = [parent]
    
        # Location Rating Inference for Similarity
        simularity = {}
        u_entropy = {}
        u_count = self.uv_cnt.loc[u].sum()  # Count of all the visits a user has made
    
        if debug:
            print 'Calculating Simularity Amongst Experts...'
        progress = float(len(candidate_experts))
        for prog, expert in enumerate(candidate_experts):
            if debug:
                if prog % (progress/10) == 0:
                    print '    Progress:', prog*100/progress, '%', datetime.now()
    
            # Level sum and Entropy
            level_sum = {}
            entropy = {}
            count = self.uv_cnt.loc[expert].sum()
    
            expert_pref = [e_pref for e_pref in self.ucw[expert] 
                           if self.ucw[expert][e_pref] > 0]
            overlapped_cates = np.intersect1d(
                pref, expert_pref, assume_unique=True
            )
            for node in overlapped_cates:
                if node not in self.h.keys():
                    continue
                # The level which a category belongs on is defined by their parent
                parent = self.h[node]['parent']

                expert_w = self.ucw[expert][node]
                u_wc = self.ucw[u][node]
                if parent in level_sum:
                    level_sum[parent] += min(expert_w, u_wc)
                else:
                    level_sum[parent] = min(expert_w, u_wc)
    
                idxs = self.venues[node]
                c_count = self.uv_cnt.loc[expert][idxs].sum()
                u_c_count = self.uv_cnt.loc[u][idxs].sum()
    
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
            simularity[expert] = [0]  # make it an array for Pandas reasons
            for level in level_sum.keys():
                simularity[expert][0] += \
                    self.h.get_level_num(level)*level_sum[level] / \
                    (1+abs(u_entropy[level] - entropy[level]))
    
        # Location Rating Calculation
        v = self.uv_cnt.loc[simularity.keys()]
        venue_ids = v.columns
        sim = pd.DataFrame.from_dict(simularity)
        r = np.dot(sim.values,v.values)
    
        # get top N recommendations from r
        if r.size == 0:
            return []
        rec = zip(venue_ids.values.tolist(), r[0].tolist())
        rec = sorted(rec, key=lambda x: x[1], reverse=True)
        top = rec[:n_recs]
        return top
