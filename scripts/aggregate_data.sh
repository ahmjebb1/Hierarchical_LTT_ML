#!/bin/bash
#loops over every file in the current directory
#takes file name x extracts the first five characters and stores as exp
#reads file and takes only lines containing "Test-set"
#the whole output produces a CSV-style summary across all experiment runs
for x in `ls` ; do exp=`echo $x | cut -c1-5` ; cat $x | grep "Test-set" | awk -v fn="$exp" '{print fn","$4","$6}' ; done

