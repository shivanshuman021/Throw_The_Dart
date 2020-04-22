import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from plot import *
from model import get_model


## Model

model = get_model()


## Data

darts = pd.read_csv("darts.csv")
# convert darts['competitor'] to a categorical variable
darts['competitor'] = pd.Categorical(darts['competitor'])
#label encoding
darts['competitor'] = darts['competitor'].cat.codes
coordinates = np.array(darts.drop(['competitor'],axis=1))
competitors = np.array(to_categorical(darts['competitor']))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


y = np.array(darts.iloc[:,[-1]])
coord_train,coord_test,competitor_train,competitor_test = train_test_split(coordinates,competitors,test_size = 0.3)
actual = [np.argmax(act) for act in competitor_test]

## Data Visualization
plot(coordinates[:,0],coordinates[:,1],title = "Darts-Distribution",cc=np.ravel(y))


model.fit(coord_train,competitor_train,epochs = 200)

accuracy  = model.evaluate(coord_test,competitor_test)
print("\n\nAccuracy = "+str(accuracy[1])+"\n\n")
preds = model.predict(coord_test)

# Print preds vs true values
#print("{:45} | {}".format('Raw Model Predictions','True labels'))


'''
for i,pred in enumerate(preds):
    print("{} | {}".format(pred,competitor_test))
'''



# Extract the indexes of the highest probable predictions

preds = [np.argmax(pred) for pred in preds]

plot(coord_test[:,0],coord_test[:,1],"Test-set actual dart-location",actual)
plot(coord_test[:,0],coord_test[:,1],"Test-set predictions",preds)



#print(preds)
#print("{:10} | {}".format('Rounded Model Predictions','True labels'))


'''
for i,pred in enumerate(preds):
    print("{:25} | {}".format(pred,competitor_test[i]))
'''
