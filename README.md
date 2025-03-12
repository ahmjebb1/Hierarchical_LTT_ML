# ltt-ml
Long-tailed Tit Call Machine Learning Project

# data
- `scripts/datasort.py` - use this to read through all of the raw selections/images files produced by the Lotti app and build pickles containing the calls, one pickle per bird (class).
- `scripts/get_LTT_class_sizes.py` - point this to the LTT files produced above - it will print out a list with the length of each class
- `scripts/reduce_GTc_to_GTr.py` - edit this so it's list of class sizes is accurate (paste the output of above in the right place) and it will produce a great tit dataset the same shape as the LTT dataset
