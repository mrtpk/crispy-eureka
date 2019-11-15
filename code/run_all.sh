#!/usr/bin/env bash
usage() {                                      # Function: Print a help message.
  echo "Usage: $0 [ -c ID_CUDA_DEVICE ] [-p CONFIG_PATH]" 1>&2
}

exit_abnormal() {                              # Function: Exit with error.
  usage
  exit 1
}
re_isanum='^[0-9]+$'
optspec=":hc:p:"
CONFIG_PATH='../config/experiment0/'
while getopts "$optspec" option; do
  case "${option}" in
    h)
      exit_abnormal
      ;;
    c)
      ID_CUDA_DEVICE=${OPTARG}
      ;;
    p)
      CONFIG_PATH=${OPTARG}
  esac
done

if ! [[ $ID_CUDA_DEVICE =~ $re_isanum ]] ; then
  echo "Error: CUDA must be a positive, whole number."
  exit_abnormal
  exit 1
fi

# look for all the json files inside CONFIG_PATH
filenames=`find $CONFIG_PATH -maxdepth 1 -name "*.json"`

for file in $filenames
do
  echo $file
  python train.py --cuda_device=$ID_CUDA_DEVICE --config_file=$file
done