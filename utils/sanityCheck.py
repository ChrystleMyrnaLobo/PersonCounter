import pandas as pd
import os
import csv
import numpy as np
import argparse
import configparser

class SanityChecker:
    """ Check if results are sane :P """
    def __init__(self):
        pass

    def str(self):
        txt = "Sanity checker \n"
        return txt

    def checkOutput(self, ipf):
        """" Parse the csv Return if Invalid """
        header = ['frame_id', 'phase', 'local_id', 'x', 'y', 'w', 'h', 'lag', 'global_id']
        dt = pd.read_csv(ipf, names=header)
        for mask in [ (dt.x < 0), (dt.y < 0) ]:
            cnt =  dt.loc[mask].shape[0]
            if cnt > 0:
                print("Redo {}".format(ipf))
                return 1
        return 0
        # print("Good {}".format(ipf))

if __name__ == '__main__' :
    # python tracker_person_counter.py -v MOT16-10 -dh ~/4Sem/MTP1/MOT16
    # 2>&1 | tee output/log.txt # Stream log to file
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--input_dir", type=str, default="/home/chrystle/4Sem/MTP1/PersonCounter/output/localtrack", help="path to log home")

    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        print(str(key) + ': ' + str(value))

    sc = SanityChecker()
    cnt = 0
    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.csv'):
            continue
        ipf = os.path.join(args.input_dir, filename)
        cnt += sc.checkOutput(ipf)
    print("Count : {}".format(cnt))
