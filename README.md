# Basic Person Counter
Keep track of people entering and leaving the frame

### Data set
MOT 16 https://motchallenge.net/data/MOT16/

### Directory Structure
```
MOT16
/train
  /MOT16-02
    /seqinfo.ini
    /img1
    /gt
        gt.txt

PersonCounter
 /Output
   /ModelA
        prediction                 // Pickle file of groundtruth and prediction
        /Image                     // Folder of images with GT and/or predicted BB
        evaluate                   // Results of evalute in a csv file
 person_counter.ipynb
```
