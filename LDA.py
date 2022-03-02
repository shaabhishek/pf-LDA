import argparse

from utils import visualize_topics, check_and_load_data_LDA


def main(args):
    _check_and_load_data = lambda split: check_and_load_data_LDA(split, args.dataroot, args.dataname, args.batch_size)

    loader_train, loader_val, loader_test = list(map(_check_and_load_data, ['train', 'val', 'test']))

    def fit_LDA_sklearn():
        from sklearn.decomposition import LatentDirichletAllocation
        model = LatentDirichletAllocation(n_components=args.K, verbose=1, max_iter=30, n_jobs=3,
                                          learning_method='online')
        model.fit(loader_train.dataset.W)
        topics = model.components_
        normalized_topics = topics / topics.sum(-1, keepdims=True)
        # out = f"topics: \n{topics.round(2)} \n"
        out = f"topics: \n{normalized_topics.round(2)} \n"
        visualize_topics(topics, "LDA-sklearn.pdf")
        print(out)

    fit_LDA_sklearn()


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument("--K", type=int, default=2, help="Number of topics")
        # parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs to train")
        # parser.add_argument("--lr", type=float, default=1e-1, help="Initial learning rate")
        parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
        parser.add_argument("--dataroot", type=str,
                            default="/Users/abhisheksharma/PycharmProjects/pfLDA/data/simulated",
                            help="Root path of data splits to which dataname_{train, val, test}.pkl are appended")
        parser.add_argument("--dataname", type=str, default="simple",
                            help="Data name where DATANAME_{train, val, test}.pkl are the data files")

        args = parser.parse_args()

        main(args)
