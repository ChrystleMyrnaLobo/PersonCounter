import argparse
import basic_person_counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=int,nargs='?', default=2)
    parser.add_argument('--video',type=int, nargs='?', default=10)
    parser.add_argument('--useGT',type=int, nargs='?', default=1)
    parser.add_argument('--iou',type=float, nargs='?', default=0.3)
    parser.add_argument('--stepSize',type=float, nargs='?', default=1)

    args = parser.parse_args()
    print "gt_Iou", "Person count"
    bpc = basic_person_counter.BasicPersonCounter(args.useGT, args.video, args.model, args.iou, args.stepSize)
    bpc.assign_id()
    print args.iou, bpc.person_counter

    # for iou in [0.1, 0.2, 0.3, 0.5, 0.7]:
    #     bpc = basic_person_counter.BasicPersonCounter(args.useGT, args.video, args.model, iou)
    #     bpc.assign_id()
    #     print iou, bpc.person_counter
    print "Done"

if __name__ == "__main__":
    main()
