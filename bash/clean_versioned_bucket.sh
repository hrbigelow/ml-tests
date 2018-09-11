#/bin/bash

# This script needs a utiilty called jq (json query)
# This also assumes aws configure output is set to json (4th choice in 'aws configure')
# The s3api delete-objects is limited to 1000 objects per request
# It seems that the jq step is a limiting factor here

bucket=$1
catalog_file=$2 # produce this using bucket_object_catalog.sh
quiet=$3
tmpidsfile=$(mktemp XXXXX.ids.json)


for field in Versions DeleteMarkers;
do
    nobj=$(cat $catalog_file | jq '.'$field' | length')
    jq_cmd=$(printf '.%s | .[$i:$j] | ' $field)
    jq_cmd+=$(printf '{ Objects: map({Key: .Key, VersionId: .VersionId}), Quiet: %s }' $quiet)
    echo 'Using jq command: ' $jq_cmd
    echo 'Deleting ' $nobj ' objects'
    for i in $(seq 0 1000 $nobj);
    do
        j=$((i+1000))
        echo 'Deleting ' $i ' to ' $j 
        cat $catalog_file | jq --argjson i $i --argjson j $j "$jq_cmd" > $tmpidsfile
        aws s3api delete-objects --bucket $bucket --delete file://$tmpidsfile
    done
done

rm $tmpidsfile

