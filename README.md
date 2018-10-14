# Basic Person Counter
Keep track of people entering and leaving the frame using IoU for data association

### Data set
MOT16 dataset obtained from [website] [paper]

### Directory Structure
Dataset
```
MOT16
/train
  /MOT16-02
    /seqinfo.ini
    /img1
    /gt
        gt.txt
```
Output
```
PersonCounter
/Output
   /ModelA_dataset
        filtered_prediction             // Pickle file of prediction only for person class
        dt_IoUx
                /Image                  // Folder of images with GT and/or predicted BB
                result_frame.csv        // Detection per frame per MOT format
                result_person.csv       // Detection per object per frame as per MOT format
```

[website]: https://motchallenge.net/data/MOT16/
[paper]: https://arxiv.org/pdf/1603.00831.pdf
