# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:24:25 2015

@author: nhatbui
"""
from collections import OrderedDict
import json


def _create_hierarchy(hierarchy, node, parent):
    for category in node['categories']:
        if 'id' in category:
            hierarchy.add_node(category['id'])
            if parent:
                hierarchy.add_edge(parent, category['id'])
        if 'categories' in category:
            _create_hierarchy(hierarchy, category, category['id'])

def _create_hierarchy_with_names(hierarchy, node, parent):
    for category in node['categories']:
        if 'id' in category:
            hierarchy.add_node(category['shortName'])
            if parent:
                hierarchy.add_edge(parent, category['shortName'])
        if 'categories' in category:
            _create_hierarchy_with_names(hierarchy, category, category['shortName'])


def load(name=False):
    h = CategoryHierarchy()
    with open('/Users/nhatbui/Documents/dev/238proj/data/category_hierarchy.json',
              'rb') as f:
                  o = json.load(f)
    categories = o['response']
    if 'categories' in categories:
        if name:
            _create_hierarchy_with_names(h, categories, None)
        else:
            _create_hierarchy(h, categories, None)
    else:
        raise Exception('A node does not have categories')
    return h


class CategoryHierarchy:
    def __init__(self):
        self.hierarchy = OrderedDict()
        
    def add_node(self, id):
        self.hierarchy[id] = { "children": [], "parent": None }
        
    def add_edge(self, from_id, to_id):
        if from_id not in self.hierarchy:
            self.add_node(from_id)
        self.hierarchy[from_id]["children"].append(to_id)
        if to_id not in self.hierarchy:
            self.add_node(to_id)
        self.hierarchy[to_id]["parent"] = from_id
        
    def keys(self):
        return self.hierarchy.keys()
        
    def __getitem__(self, key):
        return self.hierarchy[key]

    def __contains__(self, item):
        return item in self.hierarchy
        
    def __len__(self):
        return len(self.hierarchy)
        
    def get_level_num(self, category):
        if category in self.hierarchy:
            level = 0
            parent = self.hierarchy[category]['parent']
            while parent:
                level += 1
                parent = self.hierarchy[parent]['parent']
            return level
        else:
            return 0
            
    # level distance to Lowest Common Ancestor
    def dist_to_LCA(self,n1, n2, l):
        if n1 == n2:
            return l
        else:
            l1 = self.get_level_num(n1)
            l2 = self.get_level_num(n2)
            if l1 < l2:
                # l2 is farther away from the node.
                return self.dist_to_LCA(n1, self.hierarchy[n2]['parent'], l+1)
            elif l1 > l2:
                # l1 is farther away from the node.
                return self.dist_to_LCA(self.hierarchy[n1]['parent'], n2, l+1)
            else:
                # they are on the same level but do not have the same parent
                return self.dist_to_LCA(self.hierarchy[n1]['parent'],
                                        self.hierarchy[n2]['parent'],
                                        l+1)
