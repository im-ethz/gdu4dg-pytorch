import traceback

from sklearn.model_selection import ParameterGrid

from torch.utils.data import DataLoader

from config import args
from data import DigitFiveDataset
from read import load_digit5

def digits_classification(TARGET_DOMAIN, data_dir,
                          SOURCE_SAMPLE_SIZE,
                          TARGET_SAMPLE_SIZE,
                          similarity,
                          single_best = False,
                          single_source_domain = None,
                          batch_size = 128,
                          batch_norm = False,
                          lr = 0.001,
                          activation = 'tanh',
                          dropout = 0.5,
                          optimizer = 'Adam',
                          epochs = 100,
                          patience = 10,
                          early_stopping = True,
                          bias = False,
                          fine_tune = True,
                          lambda_sparse = 0,
                          lambda_OLS = 0,
                          lambda_orth = 0,
                          kernel = None,
                          num_domains = 5,
                          domain_dim = 10,
                          sigma = 7.5,
                          softness_param = 2,
                          save_file = True,
                          save_plot = False,
                          save_feature = False,
                          **kwargs):

    # read data
    data = load_digit5(TARGET_DOMAIN, data_dir, SOURCE_SAMPLE_SIZE, TARGET_SAMPLE_SIZE)

    data_train = DigitFiveDataset(data, 'train')
    data_test = DigitFiveDataset(data, 'test')

    train_loader = DataLoader(data_train, batch_size=int(1e10), shuffle=True)
    test_loader = DataLoader(data_test, batch_size=int(1e10), shuffle=True)

    # reg method is SRIP if similarity = projected, else none
    SOURCE_DOMAINS = ('mnist', 'mnistm', 'svhn', 'syn', 'usps')


if __name__ == '__main__':
    print(args)

    # define hyperparameters
    hyperparams = {k:v for k, v in vars(args).items() if type(v) == list}
    print("Hyperparams: ", hyperparams)
    grid = ParameterGrid(hyperparams)

    for params in grid:
        experiment = {**vars(args), **params}
        print(experiment)
        try:
            digits_classification(**experiment)
        except:
            traceback.print_exc()
            pass