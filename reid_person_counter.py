from utils.dataset_reid import MOT16_reid

def main():
    ds = MOT16_reid(10,True)
    ds.createGalleryAndProbe()

if __name__ == "__main__":
    main()
