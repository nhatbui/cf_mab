# Location-based Recommendation System
This project contains implementations of a collaborative filter and a multi-armed bandit.

* [learners/cf.py](https://github.com/nhatbui/cf_mab/blob/master/learners/cf.py): This code implements a collaborative filter for a location-based recommendation system as described in ["Location-based and Preference-aware Recommendation using Sparse Geo-Social Networking Data"](http://research.microsoft.com/pubs/172445/LocationRecommendation.pdf).
* [learners/container_mab.py](https://github.com/nhatbui/cf_mab/blob/master/learners/container_mab.py) and [temporal_mab.py](https://github.com/nhatbui/cf_mab/blob/master/learners/temporal_mab.py): This code implements a [UCB1 Multi-Armed Bandit](http://hunch.net/~coms-4771/lecture20.pdf). The class is a collection of bandits which optimizes for the container they represent (in this case, hour of day).
* [simulate1.py](https://github.com/nhatbui/cf_mab/blob/master/simulate1.py): runs a simulation using only the collaborative filter.
* [simulate2.py](https://github.com/nhatbui/cf_mab/blob/master/simulate2.py): runs a simulation using the collaborative filter and multi-armed bandit in parallel.
* [categories_hierarchy.py](https://github.com/nhatbui/cf_mab/blob/master/categories_hierarchy.py): create an in-memory representation of the Foursquare venues categories hierarchy.
* [data/categories_hierarchy.json](https://developer.foursquare.com/categorytree): json of categories hierarchy obtained from [Foursquare's API](https://developer.foursquare.com/docs/venues/categories).

The code is implemented in Python.

[Data to run the simulations](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) should be obtained from [Dingqi Yang](https://sites.google.com/site/yangdingqi/).

My findings are summarized in this [paper](https://www.dropbox.com/s/mcwk39u5zdjdc56/NB_FinalProject_Paper.pdf?dl=0). I used the New York and Tokyo dataset.
