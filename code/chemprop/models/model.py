from argparse import Namespace

import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout, RegularizationAccumulator



class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, aleatoric: bool, epistemic: str):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

        self.aleatoric = aleatoric
        self.epistemic = epistemic
        self.mc_dropout = self.epistemic == 'mc_dropout'

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
            ]
            last_linear_dim = first_linear_dim
        else:
            ffn = [
                dropout,
                ConcreteDropout(layer=nn.Linear(first_linear_dim, args.ffn_hidden_size),
                                reg_acc=args.reg_acc, weight_regularizer=wd,
                                dropout_regularizer=dd) if self.mc_dropout else
                nn.Linear(first_linear_dim, args.ffn_hidden_size)

            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    ConcreteDropout(layer=nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                                    reg_acc=args.reg_acc, weight_regularizer=wd,
                                    dropout_regularizer=dd) if self.mc_dropout else
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)
                ])
            ffn.extend([
                activation,
                dropout,

            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self._ffn = nn.Sequential(*ffn)

        if self.aleatoric:
            self.output_layer = nn.Linear(last_linear_dim, args.output_size)
            self.logvar_layer = nn.Linear(last_linear_dim, args.output_size)
        else:
            self.output_layer = nn.Linear(last_linear_dim, args.output_size)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        _output = self._ffn(self.encoder(*input))

        if self.aleatoric:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            # Gaussian uncertainty only for regression, directly returning in this case
            return output, logvar
        else:
            output = self.output_layer(_output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    if args.epistemic == 'mc_dropout':
            args.reg_acc = RegularizationAccumulator()

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', aleatoric=args.aleatoric, epistemic=args.epistemic)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    if args.epistemic == 'mc_dropout':
        args.reg_acc.initialize(cuda=args.cuda)

    return model
