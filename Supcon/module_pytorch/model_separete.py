#!/usr/bin/env python
# coding:utf-8
import torch
import torch.nn.functional as F
import torchvision

class Inception_v3(torch.nn.Module):
    def __init__(self, pretrained=False, use_new_func=True, freeze_key="none"): # freeze_key = ["none", "without_bn", "all"]
        super().__init__()
        if use_new_func == True:
            weights = torchvision.models.Inception_V3_Weights.DEFAULT if pretrained == True else None
            self.backbone = torchvision.models.inception_v3(weights=weights)
        else:
            self.backbone = torchvision.models.inception_v3(pretrained=pretrained)
        
        for name, parameter in self.backbone.named_parameters():
            if (freeze_key == "all") or (freeze_key == "without_bn" and "bn" not in name):
                parameter.requires_grad = False
                print("[freeze layer] {}".format(name))
        self.in_features = 2048
        self.out_features = 1024
        self.dense = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.finetune = True if freeze_key == "all" else False

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1*base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.dense.parameters(), "lr": 1.0 * base_lr},
        ]
        
        return params

    def _forward(self, x):
        x = self.backbone._transform_input(x)
        
        # N x 3 x 299 x 299
        x = self.backbone.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.backbone.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.backbone.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.backbone.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.backbone.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.backbone.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.backbone.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.backbone.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.backbone.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.backbone.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.backbone.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.backbone.AuxLogits is not None:
            if self.backbone.training:
                aux = self.backbone.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.backbone.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.backbone.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.backbone.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.backbone.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.backbone.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        aux_defined = self.backbone.training and self.backbone.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return torchvision.models.InceptionOutputs(x, aux)
        else:
            return self.backbone.eager_outputs(x, aux)
        
    def forward(self, x):
        if self.backbone.training and self.backbone.aux_logits:
            x, aux = self._forward(x)
            out = F.normalize(self.dense(x), dim=1)
            return out, aux
        else:
            x = self._forward(x)
            out = F.normalize(self.dense(x), dim=1)
            return out


class VGG19_BN(torch.nn.Module):
    def __init__(self, pretrained=False, use_new_func=True, freeze_key="none"): # freeze_key = ["none", "without_bn", "all"]
        super().__init__()
        if use_new_func == True:
            weights = torchvision.models.VGG19_BN_Weights.DEFAULT if pretrained == True else None
            self.backbone = torchvision.models.vgg19_bn(weights=weights)
        else:
            self.backbone = torchvision.models.vgg19_bn(pretrained=pretrained)
        
        for name, parameter in self.backbone.features.named_parameters():
            if (freeze_key == "all") or (freeze_key == "without_bn" and "bn" not in name):
                parameter.requires_grad = False
                print("[freeze layer] {}".format(name))
        self.in_features = 4096
        self.out_features = 1024
        self.dense = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.finetune = True if freeze_key == "all" else False

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.features.parameters(), "lr": 0.1*base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.dense.parameters(), "lr": 1.0 * base_lr},
        ]
        
        return params
        
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        out = F.normalize(self.dense(x), dim=1)
        
        return out


class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=False, use_new_func=True, freeze_key="none"): # freeze_key = ["none", "without_bn", "all"]
        super().__init__()
        if use_new_func == True:
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained == True else None
            self.backbone = torchvision.models.resnet50(weights=weights)
        else:
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        
        for name, parameter in self.backbone.named_parameters():
            if (freeze_key == "all") or (freeze_key == "without_bn" and "bn" not in name):
                parameter.requires_grad = False
                print("[freeze layer] {}".format(name))
        self.in_features = 2048
        self.out_features = 1024
        self.dense = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.finetune = True if freeze_key == "all" else False

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1*base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.dense.parameters(), "lr": 1.0 * base_lr},
        ]
        
        return params
        
    def _forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        x = self._forward(x)
        out = F.normalize(self.dense(x), dim=1)
        
        return out


class EfficientNet(torch.nn.Module):
    def __init__(self, network, pretrained=False, use_new_func=True, freeze_key="none"): # freeze_key = ["none", "without_bn", "all"]
        super().__init__()
        
        if network == "EfficientNetB0":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b0(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
            self.in_features = 1280
            self.out_features = 1024
        elif network == "EfficientNetB1":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b1(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b1(pretrained=pretrained)
            self.in_features = 1280
            self.out_features = 1024
        elif network == "EfficientNetB2":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b2(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b2(pretrained=pretrained)
            self.in_features = 1408
            self.out_features = 1024
        elif network == "EfficientNetB3":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b3(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b3(pretrained=pretrained)
            self.in_features = 1536
            self.out_features = 1024
        elif network == "EfficientNetB4":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b4(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b4(pretrained=pretrained)
            self.in_features = 1792
            self.out_features = 1024
        elif network == "EfficientNetB5":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b5(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b5(pretrained=pretrained)
            self.in_features = 2048
            self.out_features = 1024
        elif network == "EfficientNetB6":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b6(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b6(pretrained=pretrained)
            self.in_features = 2304
            self.out_features = 1024
        elif network == "EfficientNetB7":
            if use_new_func == True:
                weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT if pretrained == True else None
                self.backbone = torchvision.models.efficientnet_b7(weights=weights)
            else:
                self.backbone = torchvision.models.efficientnet_b7(pretrained=pretrained)
            self.in_features = 2560
            self.out_features = 1024
        
        for name, parameter in self.backbone.named_parameters():
            if (freeze_key == "all") or (freeze_key == "without_bn" and "bn" not in name):
                parameter.requires_grad = False
                print("[freeze layer] {}".format(name))
        self.dense = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.finetune = True if freeze_key == "all" else False
        
    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1*base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.dense.parameters(), "lr": 1.0 * base_lr},
        ]
        
        return params
        
    def _forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        x = self._forward(x)
        out = F.normalize(self.dense(x), dim=1)    
        
        return out


class Classifier(torch.nn.Module):
    def __init__(self, class_num, separate_abs_sign, in_features):
        super().__init__()

        self.in_features = in_features
        self.separate_abs_sign = separate_abs_sign
        
        if self.separate_abs_sign == 3:
            self.predictions = torch.nn.Linear(in_features=self.in_features, out_features=class_num, bias=True)
            self.softmax = torch.nn.Softmax(dim=1)
        elif self.separate_abs_sign == 1:
            self.predictions_abs = torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            self.predictions_sign = torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)
            self.sigmoid = torch.nn.Sigmoid()
        elif self.separate_abs_sign == 0:
            self.predictions = torch.nn.Linear(in_features=self.in_features, out_features=class_num, bias=True)
        else:
            self.predictions = torch.nn.Linear(in_features=self.in_features, out_features=1, bias=True)

    def get_parameters(self, base_lr=1.0):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = []
        
        if self.separate_abs_sign == 3:
            params.append({"params": self.predictions.parameters(), "lr": 1.0 * base_lr})
        elif self.separate_abs_sign == 1:
            params.append({"params": self.predictions_abs.parameters(), "lr": 1.0 * base_lr})
            params.append({"params": self.predictions_sign.parameters(), "lr": 1.0 * base_lr})
        elif self.separate_abs_sign == 0:
            params.append({"params": self.predictions.parameters(), "lr": 1.0 * base_lr})
        else:
            params.append({"params": self.predictions.parameters(), "lr": 1.0 * base_lr})

        return params
        
    def forward(self, x):
        
        if self.separate_abs_sign == 3:
            out = self.predictions(x)
            out = self.softmax(out)
        elif self.separate_abs_sign == 1:
            out_abs = self.predictions_abs(x)
            out_abs = self.relu(out_abs)
            out_sign = self.predictions_sign(x)
            out_sign = self.sigmoid(out_sign)
            out = torch.cat([out_abs, out_sign], dim=1)
        elif self.separate_abs_sign == 0:
            out = self.predictions(x)
        else:
            out = self.predictions(x)
            
        return out


def create_model_full_scratch(network, trainable_only_bn=False):
    
    freeze_key = "without_bn" if trainable_only_bn == True else "none"
    
    if network == "InceptionV3":
        model = Inception_v3(pretrained=True, use_new_func=True, freeze_key=freeze_key)
    elif network == "VGG19_bn":
        model = VGG19_BN(pretrained=True, use_new_func=True, freeze_key=freeze_key)
    elif network == "ResNet50":
        model = ResNet50(pretrained=True, use_new_func=True, freeze_key=freeze_key)
    elif "EfficientNet" in network:
        model = EfficientNet(network, pretrained=True, use_new_func=True, freeze_key=freeze_key)
    
    return model


def create_classifier_full_scratch(num_classes, separate_abs_sign, in_features):
    
    model = Classifier(num_classes, separate_abs_sign, in_features)
    
    return model

