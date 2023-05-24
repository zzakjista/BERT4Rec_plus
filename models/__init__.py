from .bert import BERTModel
from .dae import DAEModel
from .vae import VAEModel
from .ae import AEModel

MODELS = {
    BERTModel.code(): BERTModel,
    DAEModel.code(): DAEModel,
    VAEModel.code(): VAEModel,
    AEModel.code(): AEModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
