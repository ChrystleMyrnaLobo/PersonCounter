#!/bin/bash
# Usage parse_log.sh ev_mot.txt

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh pc_mot.txt"
exit
fi

cp $1 output/log/aux.txt
LOG='eval_results.csv'
cd output/log/
# Filenames
grep 'Filename' aux.txt | awk '$6 > 0 { print substr($6,1,3)}' > aux0.txt

sed -i '/INFO -/d' aux.txt
sed -i '/Filename/d' aux.txt
sed -i '/OVERALL/d' aux.txt
sed -i '/IDF1/d' aux.txt
sed -i -e 's/\s\+/,/g' aux.txt
sed -i -e 's/%//g' aux.txt

# Generating
rm $LOG
echo 'Video,IDF1%,IDP%,IDR%,Rcll%,Prcn%,GT,MT,PT,ML,FP,FN,IDs,FM,MOTA%,MOTP,ResourceSetting' > $LOG
paste -d , aux.txt aux0.txt | column -t >> $LOG

SUMMARY='../summary.csv'
rm $SUMMARY
paste -d , summary.csv $LOG >> $SUMMARY
rm aux.txt aux0.txt
