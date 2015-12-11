# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:20:23 2015

@author: nhatbui
"""
import json
import categories_hierarchy as ch


def create_hierarchy(hierarchy, node, parent):
    for category in node['categories']:
        if 'id' in category:
            hierarchy.add_node(category['id'])
            if parent:
                hierarchy.add_edge(parent, category['id'])
        if 'categories' in category:
            create_hierarchy(hierarchy, category, category['id'])
            
def create_hierarchy_with_names(hierarchy, node, parent):
    for category in node['categories']:
        if 'id' in category:
            hierarchy.add_node(category['shortName'])
            if parent:
                hierarchy.add_edge(parent, category['shortName'])
        if 'categories' in category:
            create_hierarchy_with_names(hierarchy, category, category['shortName'])


def load(name=False):
    h = ch.CategoryHierarchy()
    with open('/Users/nhatbui/Documents/dev/238proj/data/category_hierarchy.json',
              'rb') as f:
                  o = json.load(f)
    categories = o['response']
    if 'categories' in categories:
        if name:
            create_hierarchy_with_names(h, categories, None)
        else:
            create_hierarchy(h, categories, None)
    else:
        raise Exception('A node does not have categories')
    return h