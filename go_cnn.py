import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20
batch_size = 32
lr = 0.01



class GoCNN(nn.Module):
    def __init__(self, nb_channels):
        super(GoCNN, self).__init__()

        # nb channels = couleur qui doit joueur + nb_channels - 1 derniers coups
        self.conv2D = nn.Conv2d(in_channels=nb_channels, out_channels=32, kernel_size=3, padding=0)
        self.conv1D_head = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        self.conv1D_policy = nn.Conv1d(in_channels=32, out_channels=2, kernel_size=1)
        self.batch_norm2D = nn.BatchNorm2d(32)
        self.batch_norm_head = nn.BatchNorm2d(1)
        self.batch_norm_policy = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc_head1 = nn.Linear(in_features=32*9*9, out_features=256)
        self.fc_head2 = nn.Linear(in_features=256, out_features=1)
        # Jcrois que c'est ça pr le out feature mais po sûr (vecteur avec un seul 1)
        self.fc_policy = nn.Linear(in_features=32*9*9, out_features=9*9+1)
        # Résultat pour la valeu entre -1 et 1
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.value_loss = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        self.loss_function = self.value_loss + self.policy_loss + self.regularization
                
    def forward(self, x):
        # -- Convolutionnal part --#
        # Bloc 1
        x1 = self.conv2D(x)
        x1 = self.batch_norm(x1)
        x1 = self.relu(x1)
        # Bloc 2
        x2 = self.conv2D(x1)
        x2 = self.batch_norm(x2)
        # Je crois qu'il faut passer x, mais dcp pas sûr de cmt on gère les dimensions
        x2 = torch.cat((x2, x), dim=1)
        x2 = self.relu(x2)
        # Bloc 3
        x3 = self.conv2D(x2)
        x3 = self.batch_norm(x3)
        x3 = self.relu(x3)
        # Bloc 4
        x4 = self.conv2D(x3)
        x4= self.batch_norm(x4)
        x4 = torch.cat((x4, x), dim=1)
        x4 = self.relu(x4)
        # x4 = info obtenue par les couches de convolution et utilisée pr prédire 2 prochaines valeurs

        # -- Value head part --#
        value = self.conv1D_head(x4)
        value = self.batchnorm(value)  # batchnorm 2D ou 1D
        value = self.relu(value)
        value = self.fc_head1(value)
        value = self.relu(value)
        value = self.fc_head2(value)
        value = self.tanh(value)  # 1 valeur entre -1 et 1

        # -- Policy head part --#
        policy = self.conv1D_policy(x4)
        policy = self.batch_norm2D(policy)
        policy = self.relu(policy)
        policy = self.fc_policy(policy)
        policy = self.softmax(policy)

        return value, policy