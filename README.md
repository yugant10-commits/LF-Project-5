# Image Similarity

This is a program to check out the similarity between two images. 

## How to use?

Run the image_similarity.ipynb file and check out if two images are similar or not.


## Usage
Add different pictures to the pictures folder and simultaneously change the values here.
```python
img1 = load_img('pictures/cat.4003.jpg',target_size=(224,224))
img3 = load_img('pictures/cat.4161.jpg',target_size=(224,224))
```

### Analyzing the result
```bash
    Similarity score using VGG16:  -0.08551093
    Similarity score using ResNet:  -0.08076449
```
This is a cosine similarity.Note that it is a number between -1 and 1. When it is a negative number between -1 and 0, 0 indicates orthogonality and values closer to -1 indicate greater similarity. The values closer to 1 indicate greater dissimilarity.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
[Yugant] 