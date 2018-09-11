#bin/bash

bucket=$1
catalog_file=$2

echo -en 'Getting catalog of all objects...'
aws s3api list-object-versions --bucket $bucket > $catalog_file 
echo 'Done.'
echo 'Wrote objects to ' $catalog_file

