# tinyROI

## Download models & get YOLOv4 repository

```console
./download_models.sh
git clone https://github.com/WongKinYiu/PyTorch_YOLOv4.git
export PYTHONPATH=$PYTHONPATH:$PWD/PyTorch_YOLOv4
```

## Run inference

```console
python inference.py --save_results --conf_thresh 0.5
```
