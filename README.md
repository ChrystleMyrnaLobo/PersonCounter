# Person Counter
Keep track of people entering and leaving the frame using IoU for data association - naive approachCount the total people present in the video (sequence of images). For a given frame `t`,
- count of people in current frame `cnt`
- count of people not present wrt previous frame `cnt_entry`
- count people present `cnt_exit`

### Data set
MOT16 dataset obtained from [website] and [paper]

### Initial setup
- Tensorflow object detection API and pretrained models present
- MOT 16 dataset downloaded as same level as this folder
- Follow `misc/readme`

### Approach 1 : Position based (IoU)
For a basic person counter, use IoU as data association of detections between two frames.
Use groundtruth `python person_counter.py --video 10 --iou 0.3  stepSize 1 --useGT`
Use detection `python person_counter.py --video 10 --iou 0.3  stepSize 1  --model 2`
where
- `video` : video number of MOT16 dataset
- `iou` : IoU threshold for matching
- `stepSize` : frame processing rate (skips intermediate frames)
- `useGT`: use groundtruth boundary box (only for MOT16 train videos)
- `model` : model to be used for object detection

### Directory Structure
Dataset
```
MOT16
|-- train
   |-- MOT16-02
      |-- seqinfo.ini
      |-- img1
      |-- gt
         |-- gt.txt
```
Code and output
```
PersonCounter
|-- Output
   |-- ModelA_dataset
       |-- prediction                 // Detection all classes (pickle)
       |--  filtered_prediction       // Detections person class (pickle)
       |-- dt_IoUx                    // dt or gt for IoU value
          |-- Image                   // Images with BB
          |-- result_frame.csv        // Per frame
          |-- result_person.csv       // Per object per frame
```

[website]: https://motchallenge.net/data/MOT16/
[paper]: https://arxiv.org/pdf/1603.00831.pdf
