import logging
from logging.config import fileConfig
import argparse
import basic_person_counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',type=int, nargs='?', default=10, help="video number of MOT16 dataset")
    parser.add_argument('--iou',type=float, nargs='?', default=0.3, help="iou threshold for matching")
    parser.add_argument('--stepSize',type=float, nargs='?', default=1, help="processing speed fps")
    parser.add_argument('--useGT', default=False, action='store_true', help="use GT")
    parser.add_argument('--model',type=int, nargs='?', default=2, help="model for prediction")

    fileConfig('logging_config.ini')
    logger = logging.getLogger()
    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(str(key) + ': ' + str(value))

    print "gt_Iou", "Person count"
    bpc = basic_person_counter.BasicPersonCounter(args.useGT, args.video, args.model, args.iou, args.stepSize)
    # bpc.assign_id()
    print args.iou, bpc.person_counter

    # for iou in [0.1, 0.2, 0.3, 0.5, 0.7]:
    #     bpc = basic_person_counter.BasicPersonCounter(args.useGT, args.video, args.model, iou)
    #     bpc.assign_id()
    #     print iou, bpc.person_counter
    print "Done"

if __name__ == "__main__":
    main()
