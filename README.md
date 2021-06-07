# Physionet_feature_selector
apply Random forest Decision tree to extracted feature 

train_model.py > train decision tree
test_model.py > 만들어진 tree 로 2번을 수행 

1. Feature 별로 importance 값 추출 ( sklearn)
2. 만들어진 Decision Tree 의 Rule 을 기준으로 class 가 '잘' Seperate 되엇을 때, 해당 룰을 기준으로 well seperate 된 클래스를 카운트 
3. more method would be added
![image](https://user-images.githubusercontent.com/80017879/120976669-846df380-c7ad-11eb-863f-4e6fe3e6cd85.png)
