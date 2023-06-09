# Segment Anything With PaddlePaddle

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)]

![SAM design](assets/model_diagram.png?raw=true)

The **Segment Anything Model (SAM)** produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a [dataset](https://segment-anything.com/dataset/index.html) of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## Installation

Install Segment Anything:

```
pip install https://github.com/AP-Kai/segment-anything-pd.git
```

or clone the repository locally and install with

```
git clone https://github.com/AP-Kai/segment-anything-pd.git
cd segment-anything-pd; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```


## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then convert the model to paddle format by 'convert.py'.And then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="</path/to/model.pdparams>"))
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import build_sam, SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="</path/to/model.pdparams>"))
masks = mask_generator_generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```
python scripts/amg.py --checkpoint <path/to/sam/checkpoint> --input <image_or_folder> --output <output_directory>
```

See the examples notebooks on [using SAM with prompts](/notebooks/predictor_example.ipynb) and [automatically generating masks](/notebooks/automatic_mask_generator_example.ipynb) for more details.

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running 
```
from segment_anything import sam_model_registry
sam = sam_model_registry["<name>"](checkpoint="<path/to/checkpoint>")
```
Click the links below to download the checkpoint for the corresponding model name. The default model in bold can also be instantiated with `build_sam`, as in the examples in [Getting Started](#getting-started).

* **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The Segment Anything project was made possible with the help of many contributors (alphabetical):

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom
