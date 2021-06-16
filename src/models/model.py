from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.optim import Adam

def conv_dim(input_height, input_width, kernel_size, stride, padding):
    new_height = int(
        (input_height - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
    new_width = int(
        (input_width - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)

    return (new_height, new_width)


class ConvNet(nn.Module):

    def __init__(self, height=28, width=28, channels=1, classes=10, dropout=0.25):
        super(ConvNet, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.classes = classes
        self.dropout_rate = dropout

        self.conv_1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=16,
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.conv_1_dim = conv_dim(
            self.height,
            self.width,
            self.conv_1.kernel_size,
            self.conv_1.stride,
            self.conv_1.padding,
        )

        self.conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.conv_2_dim = conv_dim(
            self.conv_1_dim[0],
            self.conv_1_dim[1],
            self.conv_2.kernel_size,
            self.conv_2.stride,
            self.conv_2.padding,
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=1,
            padding=0)
        self.max_pool_dim = conv_dim(
            self.conv_2_dim[0],
            self.conv_2_dim[1],
            (self.max_pool.kernel_size, self.max_pool.kernel_size),
            (self.max_pool.stride, self.max_pool.stride),
            (self.max_pool.padding, self.max_pool.padding),
        )

        self.linear = nn.Linear(
            in_features=self.conv_2.out_channels
            * self.max_pool_dim[0]
            * self.max_pool_dim[1],
            out_features=self.classes,
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.embeddings = None
    
    def forward(self, x):
        if x.ndim not in [3, 4]:
            raise ValueError('Expected input to be a 3D or 4D tensor')

        x = self.dropout(F.relu(self.conv_1(x)))
        x = self.dropout(F.relu(self.conv_2(x)))
        x = self.dropout(self.max_pool(x))

        self.embeddings = x.reshape(-1, self.linear.in_features)
        
        x = self.linear(self.embeddings)

        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ConvNet')
        parser.add_argument('--height', type=int, default=28)
        parser.add_argument('--width', type=int, default=28)
        parser.add_argument('--channels', type=int, default=1)
        parser.add_argument('--classes', type=int, default=10)
        parser.add_argument('--dropout', type=float, default=0.25)
    
        return parent_parser
    
    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            'height': ns_dict.get('height', 28),
            'width': ns_dict.get('width', 28),
            'channels': ns_dict.get('channels', 1),
            'classes': ns_dict.get('classes', 10),
            'dropout': ns_dict.get('dropout', 0.25),
            }
        
        return args


class ImageClassifier(pl.LightningModule):

    def __init__(self, model, lr=3e-4):
        super(ImageClassifier, self).__init__()
        self.model = model
        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.save_hyperparameters()


    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss_func(preds, targets)

        return {'loss': loss, 'preds': preds, 'targets': targets}
    
    
    def training_step_end(self, outputs):
        self.train_acc(F.softmax(outputs['preds'], dim=1), outputs['targets'])
        self.log_dict(
            {'train_acc': self.train_acc, 'train_loss': outputs['loss']},
            on_step=True,
            on_epoch=True,
            prog_bar=True)
        
        return outputs


    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss_func(preds, targets)

        return {'loss': loss, 'preds': preds, 'targets': targets}


    def validation_step_end(self, outputs):
        self.val_acc(F.softmax(outputs['preds'], dim=1), outputs['targets'])
        self.log_dict(
            {'val_acc': self.val_acc, 'val_loss': outputs['loss']},
            on_step=True,
            on_epoch=True,
            prog_bar=True)


    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss_func(preds, targets)

        return {'loss': loss, 'preds': preds, 'targets': targets}


    def test_step_end(self, outputs):
        self.test_acc(F.softmax(outputs['preds'], dim=1), outputs['targets'])
        self.log_dict(
            {'test_acc': self.test_acc, 'test_loss': outputs['loss']},
            on_step=False,
            on_epoch=True,
            prog_bar=False)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ImageClassifier')
        parser.add_argument('--lr', default=3e-4, type=float)

        return parent_parser
    
    
    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            'lr': ns_dict.get('lr', 3e-4),
            }
        
        return args