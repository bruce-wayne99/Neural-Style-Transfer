from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cPickle

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images. 
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("../../images/colorful1.jpg")
content_img = image_loader("../../images/batman.jpeg")
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


######################################################################
# Now, let's create a function that displays an image by reconverting a 
# copy of it to PIL format and displaying the copy using 
# ``plt.imshow``. We will try displaying the content and style images 
# to ensure they were imported correctly.

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('output_5.png')
    #plt.pause(1) # pause a bit so that plots are updated


# plt.figure()
# imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input 

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product
    G = G.view(a, -1)
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
 

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.gram_matrix = G
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['relu4_1']
# style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 1  # increment every time we see a conv
    j = 1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv{}_{}'.format(i, j)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(i, j)
            j += 1
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}'.format(i)
            i += 1
            j = 1
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# Next, we select the input image. You can use a copy of the content image
# or white noise.
# 

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
# imshow(input_img, title='Input Image')


######################################################################
# Gradient Descent
# ----------------
# 
# As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
# L-BFGS algorithm to run our gradient descent. Unlike training a network,
# we want to train the input image in order to minimise the content/style
# losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
# our image to it as the tensor to optimize.
# 

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

with open('/scratch/pca_1000.pickle') as outfile:
    ipca_1000 = cPickle.load(outfile)

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            gram_matrices = []
            target_style = []
            for sl in style_losses:
                gram_matrices.append(sl.gram_matrix)
                target_style.append(sl.target)
            gram_matrices = torch.cat(gram_matrices, 1)
            target_style = torch.cat(target_style, 1)
            print(gram_matrices.size())
            print(target_style.size())

            # Reducing using PCA
            gram_matrices = ipca_1000.transform(gram_matrices)
            target_style = ipca_1000.transform(target_style)
            style_score = F.mse_loss(gram_matrices, target_style)
            # for sl in style_losses:
            #     style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


######################################################################
# Finally, we can run the algorithm.
# 

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')
# sphinx_gallery_thumbnail_number = 4
