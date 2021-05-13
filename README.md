## MobileDet Backbones

PyTorch implementation of MobileDet backbones introduced in [MobileDets: Searching for Object Detection Architectures for Mobile Accelerators](https://arxiv.org/abs/2004.14525v3). Ross Wightman's timm library has been used for some helper functions and inspiration for syntax style. The following are the main blocks used in the [tensorflow implementation] (https://github.com/tensorflow/models/blob/420a7253e034a12ae2208e6ec94d3e4936177a53/research/object_detection/models/ssd_mobiledet_feature_extractor.py) and their corresponding blocks in the timm library:

|Tensorflow|timm|
|_fused_conv|EdgeResidual|
|_inverterted_bottleneck|InvertedResidual|
|_inverted_bottleneck_no_expansion|DepthwiseSeparableConv|

[tucker-conv package] (https://pypi.org/project/tucker-conv/) is used for _tucker_conv