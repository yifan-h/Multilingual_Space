import argparse

from tasks import test_dbp5l

def main_func(args):
    test_dbp5l(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Space")

    # data
    parser.add_argument("--data_dir", type=str, default="/cluster/work/sachan/yifan/data/wikidata/downstream",
                        help="the input data directory.")
    parser.add_argument("--model_dir", type=str, default="/cluster/work/sachan/yifan/huggingface_models/bert-base-multilingual-cased",
                        help="The stored model directory.")
    parser.add_argument("--modelkg_dir", type=str, default="/cluster/project/sachan/yifan/projects/Multilingual_Space/tmp/mbert/final_v3.pt",
                        help="The stored model directory.")
    parser.add_argument("--model_name", type=str, default="mBERT",
                        help="The model to test: [mBERT, XLM, mBERT-KG, XLM-KG].")
    parser.add_argument("--tmp_dir", type=str, default="./tmp/checkpoints",
                        help="The stored model directory.")

    # model
    parser.add_argument("--device", type=int, default=5,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate of GCS.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of training epochs.")
    parser.add_argument("--batch_num", type=int, default=64,
                        help="number of triple samples per 1 batch")
    parser.add_argument("--neg_num", type=int, default=2,
                        help="number of negative samples")

    args = parser.parse_args()
    print(args)
    main_func(args)