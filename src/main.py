import argparse

from tasks import ki_mlkg


def main_func(args):
    ki_mlkg(args)  # MLKG integration (pretraining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Space")

    # data
    parser.add_argument("--data_dir", type=str, default="/cluster/work/sachan/yifan/data/wikidata/sub_clean",
                        help="the input data directory.")
    parser.add_argument("--model_dir", type=str, default="/cluster/work/sachan/yifan/huggingface_models/",
                        help="The stored model directory.")
    parser.add_argument("--simulate_model", type=str, default="bert-base-multilingual-cased",
                        help="multilingual LMs to analyze")

    # model
    parser.add_argument("--device", type=int, default=-1,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate of GCS.")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of training epochs.")
    parser.add_argument("--patience", type=int, default=10,
                        help="used for early stop")

    args = parser.parse_args()
    print(args)
    main_func(args)