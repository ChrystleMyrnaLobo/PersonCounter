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
   /ModelA_dataset
        filtered_prediction                 // Pickle file of prediction only for person class
        dt_IoUB
                /Image                  // Folder of images with GT and/or predicted BB
                dt.csv                  // Detection per frame per file as per MOT format
		summary.csv		// Per frame count of person entering and leaving frame 
 person_counter.ipynb
```
