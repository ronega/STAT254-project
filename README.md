# STAT254-project
Final project for STAT254: Modern Statistical Prediction and Machine Learning

# Structure
The structure of the project is following:

```bash
.
├── EDA and Other Things.ipynb
├── README.md
├── catboost
│   ├── catboost.ipynb
│   ├── catboost_csv
│   ├── catboost_other_experiments
│   └── catboost_with_air-hockey
├── data
│   ├── binary
│   ├── full
│   └── testing
├── final_report
│   ├── Presentation.pptx
│   └── latex
├── main.ipynb
└── weights
    ├── resnet50_native
    │   └── ...
    └── summary.csv
```

Folder `catboost` devoted to all experiments with catboost library. In folder `data/full` you can find the data I used for training. Each folder in `weights` corresponds to trained models. Other weights could be sent upon request due to its large memory size.

# Example

```python
from torchvision.models import alexnet, AlexNet_Weights

weights = AlexNet_Weights.DEFAULT
alexnet_model = alexnet(weights=weights)

model_params = {'BATCH_SIZE': 32,
                'EPOCHS': 50,
                'EARLY_STOP': 5}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = CNNSimple(model=alexnet_model,
                  transform=transform,
                  params=model_params,
                  name='alexnet_native') # make sure your name is correct if you want to test a model
model.test()
```
