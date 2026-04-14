mkdir -p openscene_camera && cd openscene_camera

seq 0 31 | xargs -P 4 -I {} bash -c '
split={}
file="openscene_sensor_mini_camera_${split}.tgz"
url="https://modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/master/openscene-v1.1/openscene_sensor_mini_camera/${file}"

echo "Downloading $file"
aria2c \
  -x 16 -s 16 -k 1M \
  --continue=true \
  --auto-file-renaming=false \
  -o "$file" \
  "$url"

if file "$file" | grep -qi "gzip compressed"; then
    echo "Extracting $file"
    tar -xzf "$file" && rm -f "$file"
else
    echo "Bad file: $file"
    rm -f "$file"
fi
'

mkdir -p openscene_lidar && cd openscene_lidar

seq 0 31 | xargs -P 4 -I {} bash -c '
split={}
file="openscene_sensor_mini_lidar_${split}.tgz"
url="https://modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/master/openscene-v1.1/openscene_sensor_mini_lidar/${file}"

echo "Downloading $file"
aria2c \
  -x 16 -s 16 -k 1M \
  --continue=true \
  --auto-file-renaming=false \
  -o "$file" \
  "$url"

if file "$file" | grep -qi "gzip compressed"; then
    echo "Extracting $file"
    tar -xzf "$file" && rm -f "$file"
else
    echo "Bad file: $file"
    rm -f "$file"
fi
'