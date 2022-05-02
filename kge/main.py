import argparse

from tasks import test_dbp5l, test_wk3l60

def main_func(args):
    if args.task_name == "kgc":
        test_dbp5l(args)
    else:
        test_wk3l60(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Space")

    # data
    parser.add_argument("--task_name", type=str, default="kgc",
                        help="task to do: kgc or ea")
    parser.add_argument("--data_dir", type=str, default="/cluster/work/sachan/yifan/data/wikidata/downstream",
                        help="the input data directory.")
    parser.add_argument("--model_dir", type=str, default="/cluster/work/sachan/yifan/huggingface_models/bert-base-multilingual-cased",
                        help="The stored model directory.")
    parser.add_argument("--modelkg_dir", type=str, default="/cluster/project/sachan/yifan/projects/Multilingual_Space/tmp/mbert_adapter",
                        help="The stored model directory.")
    parser.add_argument("--model_name", type=str, default="mBERT",
                        help="The model to test: [mBERT, XLM, mBERT-KG, XLM-KG].")
    parser.add_argument("--tmp_dir", type=str, default="./tmp/checkpoints",
                        help="The stored model directory.")

    # model
    parser.add_argument("--device", type=int, default=7,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-8,
                        help="learning rate of FT.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay value")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6,
                        help="Adam epsilon")
    parser.add_argument("--epoch", type=int, default=10,
                        help="number of training epochs.")
    parser.add_argument("--batch_num", type=int, default=8,
                        help="number of triple samples per 1 batch")
    parser.add_argument("--neg_num", type=float, default=1,
                        help="number of negative samples")
    parser.add_argument("--patience", type=int, default=2,
                        help="used for early stop")

    args = parser.parse_args()
    print(args)
    main_func(args)
