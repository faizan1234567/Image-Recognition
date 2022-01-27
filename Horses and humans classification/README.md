
In this repo, horses and human dataset has been downloaded and preprocessed. The dataset has 1283 images, 1027 images for training and rest of the images for validation. The dataset has two classes humans and horses. The dataset is small to train a model on, to make it large and diverse, data augmentation has been implemented. For this demo, [inceptionv3](https://arxiv.org/pdf/1512.00567) model has been used. The model final layer has been customized for this data. The model's earlier layers have been freezed. The model has been trained on final layer. Transfer learning and data augmentation have increased model's performance on this small dataset. The model achieved 99% validation accuracy on this dataset. 

It is pretty robust to make decision on new humans and horses datasets. 
