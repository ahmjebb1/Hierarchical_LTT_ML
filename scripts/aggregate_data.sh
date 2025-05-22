#!/bin/bash
for x in `ls` ; do exp=`echo $x | cut -c1-5` ; cat $x | grep "Test-set" | awk -v fn="$exp" '{print fn","$4","$6}' ; done

