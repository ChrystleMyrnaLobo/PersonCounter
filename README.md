# Person Counter
Count the total people present in the video (sequence of images). For a given frame, count of people in current frame, maintain the identify in further frames

### Data set
MOT16 dataset obtained from [website] and [paper]

### Initial setup
- MOT 16 dataset downloaded at `$MOT16`
- Install Tensorflow Object Detection API used for IoU
- Refer `misc` folder for auxillary files

### Approach 1 : Position based (IoU)
For data association between two frames, use IoU of bb  
Use groundtruth bb `python person_counter.py --video 10 --iou 0.3  stepSize 1 --useGT`  
Use detection bb `python person_counter.py --video 10 --iou 0.3  stepSize 1  --model 2`  
where
- `video` : video number of MOT16 dataset
- `iou` : IoU threshold for matching
- `stepSize` : frame processing rate (skips intermediate frames)
- `useGT`: use groundtruth boundary box (only for MOT16 train videos)
- `model` : model to be used for object detection

### Approach 2 : Detect and track (OpenCV)
- A Detect-and-Track simulation framework where detection is done periodically and subsequent frames (window size) are tracked.   
Eg: Simulation on tracking window of size 3 frames `[D T T T] [D T T T] [D T T T]`
- Data association via IoU is performed on between windows
- OpenCV Trackers [multiple object tracker]

#### Usage
Conda environment file `misc/cv.yml`
1. Run simulation
```
python tracker_person_counter.py -v MOT16-10 -dh $MOT16 --tracker csrt 2>&1 | tee output/log/simulation.log
```
where
  - `dataset_home`: path to dataset home
  - `video` : video stream. e.g: MOT16-10
  - `tracker` : Choose from csrt, kcf, boosting, mil, tld, medianflow, mosse
  - `window_size` : window size (#frames) for tracking
  - `detect_speed`: detection speed (sec)
  - `irs`: infinite resource setting (do not skip any frames). Default resource constraint setting  
Note: Special cases
  - Infinite resource setting: No frame are skipped. Use `-irs` flag which sets detection speed and tracking speed to 0
  - No tracking: Perform only detection and data association via IoU. Use `--window_size 0`
2. Data association
Generate global person id from detect-track paths
```
python pc_data_association.py -v MOT16-10 -dh $MOT16 -i output/local_track -o output/global_track
```
where
  - `dataset_home`: path to dataset home
  - `video` : video stream. e.g: MOT16-10
  - `input_dir` : directory of detections having local id
  - `output_dir`: detection to save detections after assigning global id
3. Evaluation  
Multiple object tracking metrics to evaluate the simulations.
```
python pc_mot_evaluate.py -dh $MOT16 2>&1 | tee output/log/mot_evaluate.log
./parse_log.sh output/log/mot_evaluate.log
```
- Setup mot metric `pip install motmetrics`
- Read about [ID measures]

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
Output
```
PersonCounter
|-- Output
   |-- local_track     // csv files having detect and track
   |-- global_track    // csv files after data association
   |-- log             // log files of the execution
   |-- data            // dataset structure with filtered GT, for pymotmetric evaluation
   summary.csv         // consolidated results from simulation
```

#### Convert frames to video
Convert sequence of images to video using [ffmpeg]  
`ffmpeg -framerate 7 -f image2 -i Frame_%03d.jpg ../output.mp4`


[website]: https://motchallenge.net/data/MOT16/
[paper]: https://arxiv.org/pdf/1603.00831.pdf
[multiple object tracker]: https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/
[ffmpeg]: https://askubuntu.com/a/610945
[py-motmetric]:https://github.com/cheind/py-motmetrics
[object tracking]:(https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/)
[ID measures]:http://vision.cs.duke.edu/DukeMTMC/IDmeasures.html
