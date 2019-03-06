import pandas as pd
import os

import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger()
logger.setLevel(20) # ignore less than level # Info 20 debug 10

def evaluate_mot(ipf,opf):
    """
    The results for each sequence must be stored in a separate .txt file in the archive's root folder. The file name must be exactly like the sequence name (case sensitive).

    The file format should be the same as the ground truth file, which is a CSV text-file containing one object instance per line. Each line must contain 10 values:

    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    # Read file.
    # header = ['frame_id', 'phase', 'local_id', 'x', 'y', 'w', 'h', 'lag', 'global_id']
    dt = pd.read_csv(ipf)

    # Convert to mot submission format
    dt = dt.rename(index=str, columns={"x": "bb_left", "y": "bb_top"})
    dt['conf'] = 1
    dt['x'] = -1
    dt['y'] = -1
    dt['z'] = -1
    res = dt[['frame_id', 'global_id', 'bb_left', 'bb_top', 'w', 'h','conf','x','y','z']]
    res = res.sort_values(by=['global_id'])
    # Save file
    res.to_csv(opf, index=False, header=None)
    # Run evaluation
    os.system('python -m motmetrics.apps.eval_motchallenge output/data output/data')

if __name__ == '__main__' :
    # python pc_evaluate.py  2>&1 | tee output/ev_mot.txt
    # vid = "MOT16-10"
    # filename = vid + "_kcf_pfr0.1_ws5.csv"
    # ipf = "output/globalda/" + filename
    # opf = "output/data/" + vid + ".txt"
    # evaluate_mot(ipf, opf)

    vid = "MOT16-10"
    input_dir = "output/globalda"
    opf = "output/data/" + vid + ".txt"
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        logger.info("Filename {}".format(filename))
        ipf = os.path.join(input_dir, filename)
        evaluate_mot(ipf, opf)
