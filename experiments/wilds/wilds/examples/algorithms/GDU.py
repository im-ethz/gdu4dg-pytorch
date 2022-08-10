import torch
import os

from .single_model_algorithm import SingleModelAlgorithm
from ..models.initializer import initialize_model
from ..utils import move_to, load

from .gdu_pytorch.model import LayerModel

urls_for_fe = {('rxrx1',0): 'https://worksheets.codalab.org/rest/bundles/0x7d33860545b64acca5047396d42c0ea0/contents/blob/rxrx1_seed%3A0_epoch%3Abest_model.pth',
               ('rxrx1',1): 'https://worksheets.codalab.org/rest/bundles/0xaf367840549942f79b5dd62a27b1f371/contents/blob/rxrx1_seed%3A1_epoch%3Abest_model.pth',
               ('rxrx1',2): 'https://worksheets.codalab.org/rest/bundles/0x21228e26705c4e05a1059de25458d2a0/contents/blob/rxrx1_seed%3A2_epoch%3Abest_model.pth',
               ('fmow',0): 'https://worksheets.codalab.org/rest/bundles/0x20182ee424504e4a916fe88c91afd5a2/contents/blob/fmow_seed%3A0_epoch%3Abest_model.pth',
               ('fmow',1): 'https://worksheets.codalab.org/rest/bundles/0x58b4aca2660d455eb74339db95e140c1/contents/blob/fmow_seed%3A1_epoch%3Abest_model.pth',
               ('fmow',2): 'https://worksheets.codalab.org/rest/bundles/0x55adb69b3ac3482393e0697a52555acf/contents/blob/fmow_seed%3A2_epoch%3Abest_model.pth',
               ('iwildcam',0): 'https://worksheets.codalab.org/rest/bundles/0xc006392d35404899bf248d8f3dc8a8f2/contents/blob/best_model.pth',
               ('iwildcam',1): 'https://worksheets.codalab.org/rest/bundles/0xe3ae2fef2d624309b40c9c8b24ca59ca/contents/blob/best_model.pth',
               ('iwildcam',2): 'https://worksheets.codalab.org/rest/bundles/0xb16de89752ec43b0bf79b36c0e6dc277/contents/blob/best_model.pth',
               ('camelyon17',0): 'https://worksheets.codalab.org/rest/bundles/0x6029addd6f714167a4d34fb5351347c6/contents/blob/best_model.pth',
               ('camelyon17',1): 'https://worksheets.codalab.org/rest/bundles/0xb701f5de96064c0fa1771418da5df499/contents/blob/best_model.pth',
               ('camelyon17',2): 'https://worksheets.codalab.org/rest/bundles/0x2ce5ec845b07488fb3396ab1ab8e3e17/contents/blob/best_model.pth',
               ('camelyon17',3): 'https://worksheets.codalab.org/rest/bundles/0x70f110e8a86e4c3aa2688bc1267e6631/contents/blob/best_model.pth',
               ('camelyon17',4): 'https://worksheets.codalab.org/rest/bundles/0x0fe16428860749d6b94dfb1fe9ffe986/contents/blob/best_model.pth',
               ('camelyon17',5): 'https://worksheets.codalab.org/rest/bundles/0x0dc383dbf97a491fab9fb630c4119e3d/contents/blob/last_model.pth',
               ('camelyon17',6): 'https://worksheets.codalab.org/rest/bundles/0xb7884cbe61584e80bfadd160e1514570/contents/blob/best_model.pth',
               ('camelyon17',7): 'https://worksheets.codalab.org/rest/bundles/0x6f1aaa4697944b24af06db6a734f341e/contents/blob/best_model.pth',
               ('camelyon17',8): 'https://worksheets.codalab.org/rest/bundles/0x043be722cf50447d9b52d3afd5e55716/contents/blob/best_model.pth',
               ('camelyon17',9): 'https://worksheets.codalab.org/rest/bundles/0xc3ce3f5a89f84a84a1ef9a6a4a398109/contents/blob/best_model.pth'}

class GDU(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, None)

        if config.gdu_kwargs['FE']:

            try:
                from urllib import request
                fe_path = f'./wilds/examples/algorithms/pre_trained_FE/{config.dataset}/best_model_{config.seed}.pth'

                if not os.path.exists(fe_path):
                    print('Download Model from URL')
                    _ = request.urlretrieve(urls_for_fe[(config.dataset, config.seed)], fe_path)

                _,_ = load(model, path = fe_path, device=config.device)

                print('Successfully loaded pretrained FE from WILDS .... ')

                for param in model.parameters():
                    param.requires_grad = False

            except Exception as e:
                print('Could not load pretrained FE ...')
                print(e)

        for layer in model.children():
            if hasattr(layer, 'out_features'):
                output_size = layer.out_features

        model = LayerModel(
            device=config.device,
            task='classification',
            feature_extractor=model,
            feature_vector_size=output_size,
            output_size=d_out,
            num_gdus=config.gdu_kwargs['num_gdus'],
            domain_dim=config.gdu_kwargs['domain_dim'],
            kernel_name=config.gdu_kwargs['kernel_name'],
            sigma=config.gdu_kwargs['sigma'],
            similarity_measure_name=config.gdu_kwargs['similarity_measure_name'],  # MMD, CS, Projected
            softness_param=config.gdu_kwargs['softness_param']
        )
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.use_unlabeled_y = config.use_unlabeled_y # Expect x,y,m from unlabeled loaders and train on the unlabeled y

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        outputs = self.get_model_output(x, y_true)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        if unlabeled_batch is not None:
            if self.use_unlabeled_y: # expect loaders to return x,y,m
                x, y, metadata = unlabeled_batch
                y = move_to(y, self.device)
            else:
                x, metadata = unlabeled_batch    
            x = move_to(x, self.device)
            results['unlabeled_metadata'] = metadata
            if self.use_unlabeled_y:
                results['unlabeled_y_pred'] = self.get_model_output(x, y)
                results['unlabeled_y_true'] = y
            results['unlabeled_g'] = self.grouper.metadata_to_group(metadata).to(self.device)
        return results

    def objective(self, results, algorithm):

        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], algorithm, return_dict=False)
        if self.use_unlabeled_y and 'unlabeled_y_true' in results:
            unlabeled_loss = self.loss.compute(
                results['unlabeled_y_pred'], 
                results['unlabeled_y_true'], 
                return_dict=False
            )
            lab_size = len(results['y_pred'])
            unl_size = len(results['unlabeled_y_pred'])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:
            return labeled_loss