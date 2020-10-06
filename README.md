# Mushroom Classification


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/walkman200_fly-agarics.jpg)

*Predicting the class of mushrooms through classification - watch out for those deadly Fly Agarics. Source: 'Walkman200', freeimages.com*


## Intro
In this repo I will use classification models to identify if mushrooms are edible or poisonous. The data is an old UCI dataset from about 30 years ago - but still in good shape today and can be found on Kaggle!

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/maarten_kruit_destroying_angle.jpg)

*A very poisonous Destroying Angle mushroom. Source: maarten_kruit,  freeimages.com*


## README Outline
* Introduction 
* Project Summary
* Repo Contents
* Prerequisites
* Feature and Definitions
* Results
* Future Work
* Built With, Contributors, Authors, Acknowledgments


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/Mushrooms_by_Color.png)

*A fun color-coded bar plot of dataset mushrooms based on color.*


## Repo Contents
This repo contains the following:
* README.md - this is where you are now!
* Jupyter_Notebook.ipynb - the Jupyter Notebook containing the finalized code for this project.
* PCA_and_Label_Encoding_Notebook.ipynb - a notebook where I tried some different approaches to issues.
* LICENSE.md - the required license information.
* mushrooms.csv - the file containing the dataset in csv.
* CONTRIBUTING.md 
* Images


## Libraries & Prerequisites
These are the libraries that I used in this project.
* numpy as np
* pandas as pd
* matplotlib.pyplot as plt
* %matplotlib inline
* seaborn as sns
* folium
* datetime as dt
* from sklearn.model_selection import train_test_split 
* from sklearn.model_selection import cross_val_score
* from sklearn.model_selection import GridSearchCV
* from sklearn.metrics import accuracy_score 
* from sklearn.metrics import f1_score 
* from sklearn.metrics import confusion_matrix 
* from sklearn.metrics import classification_report
* from sklearn.metrics import confusion_matrix
* from sklearn.linear_model import LogisticRegression
* from sklearn.ensemble import RandomForestClassifier
* import xgboost as xgb


## Features
* Attribute Information: (classes: edible=e, poisonous=p)
* cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
* cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
* cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
* bruises: bruises=t,no=f
* odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
* gill-attachment: attached=a,descending=d,free=f,notched=n
* gill-spacing: close=c,crowded=w,distant=d
* gill-size: broad=b,narrow=n
* gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
* stalk-shape: enlarging=e,tapering=t
* stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
* stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
* stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
* stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
* stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
* veil-type: partial=p,universal=u
* veil-color: brown=n,orange=o,white=w,yellow=y
* ring-number: none=n,one=o,two=t
* ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
* spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
* population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
* habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/Class_and_color.png)

*Mushroom bar plot by class and color.*


## Models
These are the models that I tried in this project:
* LogisticRegression
* Random Forest
* XGBoost
* K-Nearest Neighor (KNN)
* Support Vector Model (SVC)
* Stochastic Gradient Decent


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/Class_and_habitat.png)

*Bar plot looking at different mushroom habitats and their class.*


## Conclusions
All of my models were above 90% accuracy. The Random Forest Model had the best score with KNN and SVC pretty close.

Originally all of my models were over-fitting and getting too close to 100% accuracy. I tried using PCA components to reduce the number of features but still found that the models needed less columns. I tried using Feature Selection as well but found that it wasn't working with all the similar column names from the one hot encoding.

This has been an interesting data set with a few aspects that have really challenged me - in particular the purely numerous categorical columns and the feature selection.  I looked at a few examples of this Kaggle and noticed many people used Label Encoding verses One Hot Encoding. Plus, the PCA verses Feature Selection both added a lot of time to my project and I needed to review much data science and statistical information.


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/Class_and_odor.png)

*Looking at whether mushrooms are edible or not based on odor.*


## Future Work
I would love to go find some mushrooms around my house and categorize them in order to determine if they are poisonous or not! For now I just need to work on fixing my models. I want to do move work with feature selection - especially using the mushroom identifiers that I am most familiar with (such as size, cap-color, and location). Perhaps my model would fair better (though with lower accuracy) if I took out more of the the features.


![](https://raw.githubusercontent.com/twhipple/Mushroom_Classifier_Model/master/Images/mm_ramos_basket-of-mushrooms.jpg)

*Not sure I want to go shrooming any time soon after working with this data. Source: mm ramos, freeimages.com*


## Built With:
Jupyter Notebook
Python 3.0
scikit.learn

## Contributing
Please read CONTRIBUTING.md for details

## Authors
Thomas Whipple

## License
Please read LICENSE.md for details

## Acknowledgments
Thanks to Kaggle
Donated to UCI Machine Learning 27 April 1987
