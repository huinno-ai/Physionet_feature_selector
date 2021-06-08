# Physionet_feature_selector
apply Random forest Decision tree to extracted feature 

train_model.py > train decision tree
test_model.py > 만들어진 tree 로 2번을 수행 

1. Feature 별로 importance 값 추출 ( sklearn)
2. 만들어진 Decision Tree 의 Rule 을 기준으로 class 가 '잘' Seperate 되엇을 때, 해당 룰을 기준으로 well seperate 된 클래스를 카운트 
3. more method would be added


### input
![image](https://user-images.githubusercontent.com/80017879/121106756-9a2cf880-c841-11eb-9312-ad741e782b37.png)

train_model decision Tree model : train_model.py  
python train_mode.py [datapath]

### evaluate & Feature Selection 
python test_model [modelpath] [datapath] [outputpath]  
![image](https://user-images.githubusercontent.com/80017879/121106971-0c9dd880-c842-11eb-9777-507a2537b05f.png)
