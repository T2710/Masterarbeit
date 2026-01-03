import multiprocessing as mp
from dataloader import DatasetBuilder

def main():
    builder = DatasetBuilder("edu_fineweb10B", base_dir="datasets")
    builder.build_shards()

if __name__ == "__main__":
    mp.freeze_support()  # wichtig auf Windows
    main()
