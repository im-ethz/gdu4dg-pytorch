import os
from urllib import request


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
               ('camelyon17',9): 'https://worksheets.codalab.org/rest/bundles/0xc3ce3f5a89f84a84a1ef9a6a4a398109/contents/blob/best_model.pth',
               ('ogb-molpcba',0): 'https://worksheets.codalab.org/rest/bundles/0x170a029fe8b74781ad2c4024e4128277/contents/blob/best_model.pth',
               ('ogb-molpcba',1): 'https://worksheets.codalab.org/rest/bundles/0xb56a9c1b86734c8aa1cea35a2f767427/contents/blob/best_model.pth',
               ('ogb-molpcba',2): 'https://worksheets.codalab.org/rest/bundles/0xd9bcde3bf87241fba223442883c9450f/contents/blob/best_model.pth',
               ('amazon',0): 'https://worksheets.codalab.org/rest/bundles/0xe9fe4a12856f461193018504f8f65977/contents/blob/amazon_seed%3A0_epoch%3Abest_model.pth',
               ('amazon',1): 'https://worksheets.codalab.org/rest/bundles/0xcbcb1b4c49c0486eacfb082ca22b8691/contents/blob/amazon_seed%3A1_epoch%3Abest_model.pth',
               ('amazon',2): 'https://worksheets.codalab.org/rest/bundles/0xdf5298063529413eaf06654a5f83e4db/contents/blob/amazon_seed%3A2_epoch%3Abest_model.pth',
               ('civilcomments',0): 'https://worksheets.codalab.org/rest/bundles/0x17807ae09e364ec3b2680d71ca3d9623/contents/blob/best_model.pth',
               ('civilcomments',1): 'https://worksheets.codalab.org/rest/bundles/0x0f6f161391c749beb1d0006238e145d0/contents/blob/best_model.pth',
               ('civilcomments',2): 'https://worksheets.codalab.org/rest/bundles/0xb92f899d126d4c6ba73f2730d76ca3e6/contents/blob/best_model.pth',
               ('civilcomments',3): 'https://worksheets.codalab.org/rest/bundles/0x090f8d901fad4bd7be5adb4f30e20271/contents/blob/best_model.pth',
               ('civilcomments',4): 'https://worksheets.codalab.org/rest/bundles/0x7a2e24652b8d4129bc67368864062bb4/contents/blob/best_model.pth',
               ('poverty','A'): 'https://worksheets.codalab.org/rest/bundles/0xbe3ee369fe33424abc73ecccb5f3cfb3/contents/blob/poverty_fold%3AA_epoch%3Abest_model.pth',
               ('poverty','B'): 'https://worksheets.codalab.org/rest/bundles/0x5228855f3fb7423ba6500f0a48114d81/contents/blob/poverty_fold%3AB_epoch%3Abest_model.pth',
               ('poverty','C'): 'https://worksheets.codalab.org/rest/bundles/0x631693184ba94e689b43e3fc1e2af925/contents/blob/poverty_fold%3AC_epoch%3Abest_model.pth',
               ('poverty','D'): 'https://worksheets.codalab.org/rest/bundles/0x988ac7456e3c4161bb2969f34cf5e16a/contents/blob/poverty_fold%3AD_epoch%3Abest_model.pth',
               ('poverty','E'): 'https://worksheets.codalab.org/rest/bundles/0xd4c25459067a4579a69bed46e65aa9da/contents/blob/poverty_fold%3AE_epoch%3Abest_model.pth',
               ('py150',0): 'https://worksheets.codalab.org/rest/bundles/0x4c09b1a9f1df4911b0fdea55a6bc87b6/contents/blob/best_model.pth',
               ('py150',1): 'https://worksheets.codalab.org/rest/bundles/0x95419de2da314819878528b74179d736/contents/blob/best_model.pth',
               ('py150',2): 'https://worksheets.codalab.org/rest/bundles/0x5fe52cb1cec94224b2b673780b38ff61/contents/blob/best_model.pth',
               ('globalwheat',0): 'https://worksheets.codalab.org/rest/bundles/0x2ef2728450e24f95b7b34c697d492286/contents/blob/globalwheat_seed%3A0_epoch%3Abest_model.pth',
               ('globalwheat',1): 'https://worksheets.codalab.org/rest/bundles/0x94ea65499c1b43f0892abdcf84335fa9/contents/blob/globalwheat_seed%3A1_epoch%3Abest_model.pth',
               ('globalwheat',2): 'https://worksheets.codalab.org/rest/bundles/0x59af821e2af14473aeb8b03a35a2f75c/contents/blob/globalwheat_seed%3A2_epoch%3Abest_model.pth'}

def fe_loader (config):
    fe_path = f'./wilds/examples/algorithms/pre_trained_FE/{config.dataset}/best_model_{config.seed}.pth'
    parent_fe_path = os.path.abspath(os.path.join(fe_path, os.pardir))

    if not os.path.exists(parent_fe_path):
        os.mkdir(parent_fe_path)

    if not os.path.exists(fe_path):
        print('Download Model from URL')
        if config.dataset =="poverty":
            _ = request.urlretrieve(urls_for_fe[(config.dataset, config.dataset_kwargs['fold'])], fe_path)
        else:
            _ = request.urlretrieve(urls_for_fe[(config.dataset, config.seed)], fe_path)

    return fe_path
