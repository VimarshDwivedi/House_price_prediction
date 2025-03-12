# Multi-Modal House Price Estimation

## Overview
This project presents an approach to estimate house prices using both visual and textual features. Traditional methods rely on textual attributes such as area and number of rooms, whereas this project incorporates images of houses along with their textual descriptions to improve prediction accuracy.

## Project Contributions
- Utilized a dataset provided by Eman Ahmed that includes both images and textual attributes for house price estimation.
- Implemented both machine learning and deep learning approaches to predict house prices.
- Conducted extensive experiments to compare different models and reported the results.

## Dataset
The dataset consists of 535 sample houses from California, USA, with each house represented by:
- **Visual Data**: Four images (frontal view, bedroom, kitchen, bathroom)
- **Textual Data**: Number of bedrooms, number of bathrooms, area, and zip code

## Methodology
### 1. Machine Learning Approach
Feature engineering and preprocessing were applied to both textual and visual data. Various models were trained:
- **Classic Models**: Linear Regression, Polynomial Regression, Ridge Regression, Decision Tree Regressor
- **Advanced Models**: Random Forest Regressor, Support Vector Regressor, CatBoost, XGBoost

### 2. Deep Learning Approach
A multi-channel neural network was designed to process both textual and visual data:
- **Textual Data**: Processed using a Multilayer Perceptron (MLP)
- **Visual Data**: Processed using a Convolutional Neural Network (CNN)
- **Final Output**: Combined features were passed through a dense layer to estimate the house price

## Results
- The **Random Forest Regressor** achieved the best performance among all machine learning models.
- The **Neural Network** approach, though promising, did not outperform advanced machine learning models due to limited dataset size and simple architecture.
- Feature importance analysis highlighted that **total rooms, area, and room size** significantly impact price prediction.



## Installation & Usage
### Prerequisites
- Python 3.8+
- Libraries: TensorFlow, PyTorch, Scikit-Learn, OpenCV, Pandas, NumPy, Matplotlib

### Installation
```bash
pip install -r requirements.txt
```

### Running the Project
1. **Preprocess Data**
   ```bash
   python src/preprocessing.py
   ```
2. **Train Model**
   ```bash
   python src/train.py --model random_forest
   ```
3. **Evaluate Model**
   ```bash
   python src/evaluate.py
   ```

## References
- [House Price Estimation from Visual and Textual Features](https://arxiv.org/abs/1609.08399)
- [Vision-Based Housing Price Estimation](https://www.sciencedirect.com/science/article/pii/S2667305322000217)
- [GitHub Dataset](https://github.com/emanhamed/Houses-dataset)




