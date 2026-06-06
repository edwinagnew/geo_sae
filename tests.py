# TODO - turn into actual tests
# this is just the desired functionality I want from the code

from train import Trainer, make_trainer, TrainConfig, SignedBatchTopKSAEConfig, SignedBatchTopKSAE, CoActPenalizedSAEConfig, CoActPenalizedSAE
from data import EASY_CONFIG, PAPER_CONFIG, build_zoo

zoo = build_zoo(EASY_CONFIG)
sae_cfg = SignedBatchTopKSAEConfig(zoo.d, expansion_factor=4)

train_cfg = TrainConfig(device=device)


trainer = Trainer(sae_cfg, zoo, train_cfg)


#also want to be able to compare with normal BatchTopK
trainer2 = Trainer(SignedBatchTopKSAEConfig(), zoo, train_cfg)