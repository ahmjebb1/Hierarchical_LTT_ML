# ltt-ml
Long-tailed Tit Call Machine Learning Project

# data
Raw LTT data is processed using the updated [Lotti App](https://github.com/bubblecontrol/lotti_app). Raw GT data from the [Wytham Wood dataset](https://nilomr.github.io/great-tit-hits/), hosted at OSF: https://osf.io/n8ac9/files/osfstorage. This data downloads as four partial zips, you have to concatenate these together and then "fix" them if you are using standard tools. E.g.:
```
cat song_files.zip.part1 song_files.zip.part2 song-files.zip.part3 song-files.zip.part4 >> song-files.broken.zip
zip -FF song-files.broken.zip --out song-files.zip
cd $DESTINATION
unzip $SOURCE/song-files.zip
```


- `scripts/prep_wytham_data.py` - script to convert the raw GT data to the pickle format used by the data loader (the same as LTTs)
- `scripts/datasort.py` - use this to read through all of the raw selections/images files produced by the Lotti app and build pickles containing the calls, one pickle per bird (class).
- `scripts/get_LTT_class_sizes.py` - point this to the LTT files produced above - it will print out a list with the length of each class
- `scripts/reduce_GTc_to_GTr.py` - edit this so it's list of class sizes is accurate (paste the output of above in the right place) and it will produce a great tit dataset the same shape as the LTT dataset
