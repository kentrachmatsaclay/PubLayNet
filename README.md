# PubLayNet

PubLayNet is a large dataset of document images, of which the layout is annotated with both bounding boxes and polygonal segmentations. For more information, see [PubLayNet original](https://github.com/ibm-aur-nlp/PubLayNet)

## Recent updates 

`29/Feb/2020` - Add evaluation scores.

`22/Feb/2020` - Pre-trained Mask-RCNN model in (Pytorch) are [released](maskrcnn) .


## Inference


Download trained weights here, locate it in [maskrcnn directory](maskrcnn)


- [12000x16 iterations](https://drive.google.com/open?id=1T2ciEJ7npW_aBpNrKHiUAluyk04K0AWK)
- [50000x16 iterations](https://drive.google.com/open?id=1vl3XAYbGKlv70SNPReStZQ6I0Z9v1CSW)
- [120000x16 iterations](https://drive.google.com/open?id=13fhd_SS7fLrjLrCjVpCwOYGt_SlQ_7FW)
- [161000x16 iterations](https://drive.google.com/open?id=1KNOyw_D980bvFKb8U8NPPt-NWSsWJDe6)
- [174000x16 iterations](https://drive.google.com/open?id=13fhd_SS7fLrjLrCjVpCwOYGt_SlQ_7FW)
- [200000x16 iterations](https://drive.google.com/open?id=1rJ3fowtxGIcORzIZbQe9ibHN0ORoqkLN)


Run
```
python infer.py <path_to_image>
```

## Avarage Precision in validation stages (via Tensorboard)

<img src="https://user-images.githubusercontent.com/24642166/75600546-066b6900-5ae3-11ea-9774-a0a0396e6fb1.png" width=1000>


## Example Results

<img src="./example_images/PMC4334925_00006.jpg" width=400> | <img src="./example_images/PMC538274_00004.jpg" width=400> 
:-------------------------:|:-------------------------:
**PMC4334925_00006.jpg**  | **PMC538274_00004.jpg**




## Getting data

Images and annotations can be downloaded [here](https://developer.ibm.com/exchanges/data/all/publaynet/). The training set is quite large, so two options are offered. We split the training set into 7 batches, which can be separately downloaded. Or you can also download the full set at once.

If direct download in browser is unstable or you want to download the data from the command line, you can use curl or wget to download the data.

```
curl -o <YOUR_TARGET_DIR>/publaynet.tar.gz https://dax.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz
```

```
wget -O <YOUR_TARGET_DIR>/publaynet.tar.gz https://dax.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz
```

## Annotation format

The annotation files follows the [json format of the Object Detection task of MS COCO](http://cocodataset.org/#format-data)
