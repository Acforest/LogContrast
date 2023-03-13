folders="./datasets/HDFS/ ./datasets/BGL/ ./models/ ./logs/"
for folder in $folders
do
  if [ -e "$folder" ]
  then
    echo "$folder exists"
  else
    mkdir -p "$folder"
fi
done