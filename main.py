import sklearn
import pandas as pd
import numpy as np
import pickle
import sklearn.neighbors

#load the data
data = pd.read_csv("car.data")

#data manipulation
for x,y in zip(data["buying"], range(len(data["buying"]))):
    if(x == "vhigh"): x = 3
    elif(x == "high"): x = 2
    elif(x == "med"): x = 1
    elif(x == "low"): x = 0
    data["buying"][y] = x

for x,y in zip(data["maint"], range(len(data["maint"]))):
    if(x == "vhigh"): x = 3
    elif(x == "high"): x = 2
    elif(x == "med"): x = 1
    elif(x == "low"): x = 0
    data["maint"][y] = x

for x,y in zip(data["doors"], range(len(data["doors"]))):
    if(x == "5more"): x = 5
    else: x = int(x)
    data["doors"][y] = x

for x,y in zip(data["persons"], range(len(data["persons"]))):
    if(x == "more"): x = 5
    elif(x == '4'): x = 4
    else: x = 2
    data["persons"][y] = x

lug_boot_dict = {"small": 0,
                 "med": 1,
                 "big": 2}
for x,y in zip(data["lug_boot"],range(len(data["lug_boot"]))):
    data["lug_boot"][y] = lug_boot_dict[x]

for x,y in zip(data["safety"], range(len(data["safety"]))):
    if(x == "high"): x = 2
    elif(x == "med"): x = 1
    elif(x == "low"): x = 0
    data["safety"][y] = x

class_dict = {"unacc": 0,
              "acc": 1,
              "good": 2,
              "vgood": 3}

for x,y in zip(data["class"],range(len(data["class"]))):
    data["class"][y] = class_dict[x]

labels = data["class"]
features = data.drop(axis=1,labels="class")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.2)

xtrn = np.array(x_train)
xtst = np.array(x_test)
ytrn = np.array(y_train)
ytst = np.array(y_test)

xtrn = xtrn.astype('int')
ytrn = ytrn.astype('int')
xtst = xtst.astype('int')
ytst = ytst.astype('int')

#Define and train the model
model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
model.fit(xtrn, ytrn)

#Test and score
predicted = model.predict(xtst)
for x,y in zip(ytst, predicted):
    print("Real:",x,"Predicted:",y)
score = model.score(xtst,ytst)
print("Acc:", score)

#Saving the model
if(score >= 0.95):
    with open("BestModel", "wb") as f:
        pickle.dump(model,f)
        print("Model saved.")