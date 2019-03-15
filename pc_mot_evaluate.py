import pandas as pd
from utils.dataset import MOT16
import os, argparse
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger()
logger.setLevel(30) # ignore less than level # Info 20 debug 10

def prepare_files(gt_original, gt_transformed, dt_original, dt_transformed):
    """
        The results for each sequence must be stored in a separate .txt file. The file name must be exactly like the sequence name (case sensitive).
        The file format should be the same as the ground truth file, which is a CSV text-file containing one object instance per line. Each line must contain 10 values:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Groundtruth in <op_dir>/<seq1>/gt/gt.txt
        Annotation in <op_dir>/<seq1>.txt
    """
    #1. Convert dt to mot submission format
    # header = ['frame_id', 'phase', 'local_id', 'x', 'y', 'w', 'h', 'lag', 'global_id']
    dt = pd.read_csv(dt_original)
    dt = dt.rename(index=str, columns={"x": "bb_left", "y": "bb_top"})
    dt['conf'] = 1
    dt['x'] = -1
    dt['y'] = -1
    dt['z'] = -1
    res = dt[['frame_id', 'global_id', 'bb_left', 'bb_top', 'w', 'h','conf','x','y','z']]
    res = res.sort_values(by=['global_id'])
    res.to_csv(dt_transformed, index=False, header=None)
    logger.info("Detection from {} to {}".format(dt_original, dt_transformed))

    # Some stats about the execution
    processed_frames = dt['frame_id'].unique()
    frame_list =  dt.groupby(['frame_id','phase']).size().reset_index().rename(columns={0:'count'})
    frame_list['virtual_frame_id'] = frame_list.index

    total_person_count = dt['global_id'].max()
    cnt_processed_frame = dt.frame_id.unique().shape[0] # total count of frames processed
    cnt_detected_frame = frame_list.loc[frame_list.phase=="detect"].shape[0] # total count of frames on which detection was performed
    cnt_tracked_frame = frame_list.loc[frame_list.phase=="track"].shape[0] # total count of frames on which tracking was performed
    avg_detect_rate = -1
    if cnt_detected_frame > 1:
        avg_detect_rate = frame_list.loc[frame_list.phase=="detect"]["frame_id"].rolling(window=2).apply(lambda x: x[1] - x[0], raw=True).mean()  # Average Detect and Track period i.e.average #frames skipped between detections
    avg_track_speed = 0
    if cnt_tracked_frame > 0:
        avg_track_speed = dt.loc[dt['phase']=='track','lag'].mean()

    #2. Filter out relevant GT frames
    header = ['frame_id', 'person_id', 'bb_left', 'bb_top', 'w', 'h','conf','x','y','z']
    gt = pd.read_csv(gt_original, names=header)
    res = gt[gt['frame_id'].isin(processed_frames)]
    res.to_csv(gt_transformed, index=False, header=None)
    logger.info("Annotation from {} to {}".format(gt_original, gt_transformed))
    return (total_person_count, cnt_processed_frame, cnt_detected_frame, cnt_tracked_frame, avg_detect_rate, avg_track_speed)

def evaluate_mot():
    # Run evaluation
    os.system('python -m motmetrics.apps.eval_motchallenge output/data output/data')

if __name__ == '__main__':
    # python pc_mot_evaluate.py  2>&1 | tee output/ev_mot.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("-dh", "--dataset_home", required=True, type=str, help="path to dataset home")
    parser.add_argument("-v", "--video", type=str, default="MOT16-10", help="video stream. e.g: MOT16-10")
    parser.add_argument("-i", "--input_dir", type=str, default="output/global_track", help="path to dir having global track")
    parser.add_argument("-o", "--output_dir", type=str, default="output/data", help="path to dir for motmetrics")
    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(str(key) + ': ' + str(value))

    vid = int(args.video.split('-')[1])
    gt_original = MOT16(args.dataset_home, vid).path_to_annotation_file

    summary = []
    summary_file = os.path.join( os.path.dirname(args.input_dir), "log", "summary.csv")
    header = ['filename', 'Tracker algo', 'detect speed (s)', 'average track speed (s)', 'window size (#frames)', 'person count', 'total frames processed', 'total frames detected', 'total frames tracked', 'avg detect rate (frames)']

    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.csv'):
            continue
        gt_transformed = os.path.join(args.output_dir, args.video, "gt")
        if not os.path.exists(gt_transformed):
            os.makedirs(gt_transformed)
        gt_transformed = os.path.join(gt_transformed, "gt.txt")
        dt_original = os.path.join(args.input_dir, filename)
        dt_transformed = os.path.join(args.output_dir, args.video+".txt")
        logger.info("GT {} {}".format(gt_original, gt_transformed))
        logger.info("DT {} {}".format(dt_original, dt_transformed))
        logger.warning("Filename {}".format(filename))
        total_person_count, cnt_processed_frame, cnt_detected_frame, cnt_tracked_frame, avg_detect_rate, avg_track_speed = prepare_files(gt_original, gt_transformed, dt_original, dt_transformed)
        evaluate_mot()
        (vid, algo, pfr, ws) = str(filename).split("_")
        vid = vid.split('/')[-1]
        summary.append( [filename, algo, pfr[3:], avg_track_speed, ws[2:-4], total_person_count, cnt_processed_frame, cnt_detected_frame, cnt_tracked_frame, avg_detect_rate] )
    dt = pd.DataFrame.from_records(summary, columns=header)
    dt.to_csv(summary_file, index=False)
    logger.info("Summary file {}".format(summary_file))
