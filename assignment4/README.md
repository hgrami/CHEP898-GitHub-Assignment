# Independent Analysis Part 1: Human Activity Recognition Using Smartphones

## Project Description

This project analyzes smartphone sensor data to classify physical activities. The goal is to build a machine learning model that can distinguish between six different activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) based on accelerometer and gyroscope signals from a smartphone worn at the waist.

This work is relevant to health science because accurate activity classification enables:
- Objective physical activity monitoring in clinical studies
- Fall detection systems for elderly care
- Rehabilitation progress tracking
- Sedentary behavior assessment

## Dataset

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Subjects**: 30 volunteers (ages 19-48)
- **Activities**: 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- **Features**: 561 time/frequency domain variables from accelerometer and gyroscope
- **Observations**: 10,299 total (7,352 train + 2,947 test)

The dataset was collected using the embedded accelerometer and gyroscope in a Samsung Galaxy S II. Participants performed activities while wearing the smartphone on their waist. Sensor signals were pre-processed and 561 features were extracted from fixed-width sliding windows.

## Repository Structure

```
assignment4/
├── README.md                                    # This file
├── RamirezAsturiasHector_assignment4.Rmd        # Analysis plan and data wrangling
├── RamirezAsturiasHector_assignment4.html       # Rendered output
└── data/UCI_HAR_Dataset/                        # Dataset files
    ├── activity_labels.txt                      # Activity code mappings
    ├── features.txt                             # Feature names (561)
    ├── train/                                   # Training data
    │   ├── X_train.txt                          # Feature values
    │   ├── y_train.txt                          # Activity labels
    │   └── subject_train.txt                    # Subject IDs
    └── test/                                    # Test data
        ├── X_test.txt
        ├── y_test.txt
        └── subject_test.txt
```

## How to Reproduce

1. Clone this repository
2. Ensure the UCI HAR Dataset is in `data/UCI_HAR_Dataset/`
3. Open `RamirezAsturiasHector_assignment4.Rmd` in RStudio
4. Knit to HTML

## Course Components

This assignment incorporates:
- **GitHub**: Version control and project documentation
- **R/tidyverse**: Data wrangling and visualization
- **tidymodels**: Planned for Part 2 (classification models)

## References

1. Anguita D, Ghio A, Oneto L, Parra X, Reyes-Ortiz JL. A public domain dataset for human activity recognition using smartphones. In: Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013); 2013 Apr 24-26; Bruges, Belgium. p. 437-42. Available from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Author

Hector Ramirez Asturias
CHEP 898: Machine Learning Methods in Health Science
University of Saskatchewan, Winter 2026
