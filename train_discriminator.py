import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse

# 1. VGG the backbone
def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 2. Style Extractor: extract 4 layers of feature maps in VGG (from input image)
class StyleExtractor(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, encoder):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(StyleExtractor, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:6])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[6:13])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[13:20])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[20:33])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[33:46])  # relu4_1 -> relu5_1
        self.enc_6 = nn.Sequential(*enc_layers[46:70])  # relu5_1 -> maxpool

        # fix the encoder
        for name in ['enc_1', 'enc_2','enc_3', 'enc_4', 'enc_5', 'enc_6']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # Class Activation Map
        self.conv1x1_0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True)
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.conv1x1_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(6):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, input, index):
        """Standard forward."""
        feats = self.encode_with_intermediate(input)
        codes = []
        for x in index:
            code = feats[x].clone()
            gap = torch.nn.functional.adaptive_avg_pool2d(code, (1,1))
            gmp = torch.nn.functional.adaptive_max_pool2d(code, (1,1))            
            conv1x1 = getattr(self, 'conv1x1_{:d}'.format(x))
            code = torch.cat([gap, gmp], 1)
            code = self.relu(conv1x1(code))
            codes.append(code)
        return codes

# 3. Projector: projects the style feature into a set of K-dimensional latent style code (K=2048)
class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        self.projector0 = nn.Sequential(
            nn.Linear(64, 1024),
            nn.Tanh(),
            #nn.Dropout(),
            # nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
        )
        self.projector1 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(128, 1024),
            nn.Tanh(),
            #nn.Dropout(),
            # nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
        )
        self.projector2 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256,1024),
            nn.Tanh(),
            #nn.Dropout(),
            # nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
        )
        self.projector3 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            #nn.Dropout(),
            # nn.Linear(1024, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
        )
        self.projector4 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector5 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )

    def forward(self, input, index):
        """Standard forward."""
        num = 0
        projections = []
        for x in index:
            projector = getattr(self, 'projector{:d}'.format(x))        
            code = input[num].view(input[num].size(0), -1)
            projection = projector(code).view(code.size(0), -1)
            projection = nn.functional.normalize(projection)
            projections.append(projection)
            num += 1
        return projections

# 4. Linear Regression Model: map the style code into [-1, 1]
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
  
    def forward(self, x):
        x = self.linear(x)
        return x

# 5. our StyleAuditor: Style Extractor + Projector + Regression
class StyleAuditor(nn.Module):
    def __init__(self, encoder, style_code_dim):
        super(StyleAuditor, self).__init__()
        self.style_extractor = StyleExtractor(encoder)
        self.projector = Projector()
        self.regression = LinearRegressionModel(input_dim=style_code_dim)
    
    def forward(self, input, index):
        codes = self.style_extractor(input, index)
        projections = self.projector(codes, index)
        style_code = torch.cat(projections, dim=1)
        output = self.regression(style_code)
        return output

# 6. Early stopping method
class EarlyStopping:  
    def __init__(self, patience=5, delta=0, monitor="val_loss"):  
        self.patience = patience  
        self.delta = delta  
        self.monitor = monitor
        self.counter = 0
        self.best_score = None  
        self.early_stop = False
        self.best_model = None
  
    def __call__(self, val_loss, model):  
        if self.best_score is None:  
            self.best_score = val_loss
            self.best_model = model
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_model = model
            self.counter = 0

    def get_best_model(self):
        return self.best_model

def train(train_dataset_dir, validate_dataset_dir, public_dataset_ori_dir, public_dataset_gen_dir, save_path):
    style_auditor = StyleAuditor(encoder=vgg, style_code_dim=4096)
    style_auditor.to(device)
    feature_layer_index = [0, 1, 2, 3]

    train_dataset = datasets.ImageFolder(root=train_dataset_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    validation_dataset = datasets.ImageFolder(root=validate_dataset_dir, transform=transform)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)

    public_dataset_ori = datasets.ImageFolder(root=public_dataset_ori_dir, transform=transform)
    public_dataset_gen = datasets.ImageFolder(root=public_dataset_gen_dir, transform=transform)
    public_dataset_ori_loader = DataLoader(public_dataset_ori, batch_size=15, shuffle=False)
    public_dataset_gen_loader = DataLoader(public_dataset_gen, batch_size=15, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(style_auditor.parameters(), lr=5e-5, betas=(0.5, 0.999))
    early_stopping = EarlyStopping(patience=10, delta=0.01)
    
    # train
    for epoch in range(num_epoches):
        running_loss = 0.0
        shift_loss_sum = 0.0
        anchor_loss_sum = 0.0
        optimizer.zero_grad()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # map neg=0 to neg=-1
            mask = labels == 0
            labels[mask] = -1

            outputs = style_auditor(inputs, feature_layer_index)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels.float())

            running_loss += loss


        for ori_data, gen_data in zip(public_dataset_ori_loader, public_dataset_gen_loader):
            ori_img, _ = ori_data
            gen_img, _ = gen_data
            ori_img, gen_img = ori_img.to(device), gen_img.to(device)
            labels = torch.Tensor([-1.0] * int(ori_img.shape[0])).to(device)

            ori_output = style_auditor(ori_img, feature_layer_index)
            ori_output = ori_output.squeeze()
            
            anchor_loss = criterion(ori_output, labels)
            anchor_loss_sum += anchor_loss

            ori_output = ori_output.detach()

            gen_output = style_auditor(gen_img, feature_layer_index)
            gen_output = gen_output.squeeze()

            shift_loss = criterion(ori_output, gen_output)
            shift_loss_sum += shift_loss
        
        overall_loss = running_loss + shift_loss_sum + anchor_loss_sum
        overall_loss.backward()
        optimizer.step()

        # early stopping
        validation_loss = 0
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # map neg=0 to neg=-1
            mask = labels == 0
            labels[mask] = -1

            outputs = style_auditor(inputs, feature_layer_index)
            outputs = outputs.squeeze()

            validation_loss += criterion(outputs, labels.float())

        validation_loss /= len(validation_loader)
        early_stopping(validation_loss, style_auditor)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"Epoch {epoch+1}, Train Loss: {running_loss}, Validation Loss: {validation_loss}")
        print(f"Epoch {epoch+1}, Shift Loss: {shift_loss_sum}")
        print(f"Epoch {epoch+1}, Anchor Loss: {anchor_loss_sum}")
    
    best_model = early_stopping.get_best_model()
    torch.save(best_model.state_dict(), save_path + '_best')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Auditor.")
    parser.add_argument(
        "--train_dataset_root_dir",
        type=str,
        required=True,
        help="The train dataset root dir of discriminator."
    )
    parser.add_argument(
        "--validate_dataset_root_dir",
        type=str,
        required=True,
        help="The validate dataset root dir of discriminator."
    )
    parser.add_argument(
        "--public_dataset_ori_dir",
        type=str,
        required=True,
        help="The public dataset ori root dir."
    )
    parser.add_argument(
        "--public_dataset_gen_dir",
        type=str,
        required=True,
        help="The public dataset gen root dir."
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=True,
        help="The model save dif."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        default=1,
        help="Random seed."
    )
    args = parser.parse_args()
    
    return args

args = parse_args()

# set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

imsize = 512
num_epoches = 100

transform = transforms.Compose([
    transforms.Resize(imsize, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
        512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])
vgg.to(device)
vgg.load_state_dict(torch.load('path/to/style_vgg.pth'))


if __name__ == "__main__":

    model_save_dir = args.model_save_dir + f"_seed={args.seed}"
    os.makedirs(model_save_dir, exist_ok=True)
    artist_names = sorted(os.listdir(args.train_dataset_root_dir))

    for name in artist_names:
        print(f"====== Artist: {name}")
        train_dataset_dir = os.path.join(args.train_dataset_root_dir, name)
        validate_dataset_dir = os.path.join(args.validate_dataset_root_dir, name)
        save_path = os.path.join(model_save_dir, f'artist={name}_seed={args.seed}')

        train(train_dataset_dir, validate_dataset_dir, args.public_dataset_ori_dir, args.public_dataset_gen_dir, save_path)