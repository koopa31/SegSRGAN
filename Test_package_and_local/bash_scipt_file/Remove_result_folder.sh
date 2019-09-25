#!/bin/bash


yourfilenames=$(find $1"/Image_for_testing/" -name "**" -type d )

echo "first_file:"$yourfilenames | awk '{print $0}'

for eachfile in $yourfilenames
do
   	if [[ $eachfile == *"Result"* ]]
	then
  		
	rm -rf $eachfile
	fi
done
