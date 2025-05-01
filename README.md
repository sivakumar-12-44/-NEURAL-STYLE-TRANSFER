# -NEURAL-STYLE-TRANSFER

COMPANY: CODTECH IT SOLUTIONS

NAME: BOLLAVARAM SIVA KUMAR REDDY

INTERN ID: CT04WT242

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

#This Python script implements Neural Style Transfer using PyTorch and a pre-trained VGG-19 convolutional neural network. Neural style transfer is a technique that blends the content of one image (typically a photograph) with the style of another image (such as a painting). The script loads and preprocesses input images using torchvision transforms, resizes them based on the available hardware (CUDA or CPU), and converts them to tensors. Two custom loss modules are defined: ContentLoss, which measures how different the generated image is from the content image, and StyleLoss, which compares the style (via Gram matrices) of the generated image with the style image.

A Normalization class ensures that input images are normalized according to the VGG network’s training parameters. The function get_style_model_and_losses() constructs a new model by copying the layers of VGG-19 and inserting content and style loss layers at appropriate points. The core function run_style_transfer() performs optimization on the input image, gradually updating it to minimize a combined loss of content and style, controlled by weighting factors. Finally, the script visualizes and saves the output image. The main block loads two images (a content and a style image), ensures they match in size, and then performs style transfer, outputting a stylized version of the content image. Note: there are a few typos in the code, such as using init instead of _init_ in class definitions and name instead of _name_, which need correction for the script to run properly.


OUTPUT:

![Image](https://github.com/user-attachments/assets/aa25cff4-d669-4960-9725-b79ce5cf2958)
