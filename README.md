# Physionet_feature_selector
apply Random forest Decision tree to extracted feature 

train_model.py > train decision tree
test_model.py > 만들어진 tree 로 2번을 수행 

1. Feature 별로 importance 값 추출 ( sklearn)
2. 만들어진 Decision Tree 의 Rule 을 기준으로 class 가 '잘' Seperate 되엇을 때, 해당 룰을 기준으로 well seperate 된 클래스를 카운트 
3. more method would be added


### train decision Tree sklearn
![image](https://user-images.githubusercontent.com/80017879/121107703-5b983d80-c843-11eb-8a9e-601c9d4e1d27.png)
train random forest decision tree model  
python train_mode.py [datapath] [modelpath]

### evaluate & Feature Selection 

![image](https://user-images.githubusercontent.com/80017879/121106971-0c9dd880-c842-11eb-9777-507a2537b05f.png)  
python test_model [modelpath] [datapath] [outputpath]  
