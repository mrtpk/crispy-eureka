for file in ../config/experiment0/*
do
  echo $file
  python train.py --cuda_device=2 --config_file=$file
done