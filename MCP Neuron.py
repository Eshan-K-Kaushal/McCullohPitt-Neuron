import time
import pandas as pds
import matplotlib.pyplot as pt
from sklearn.metrics import accuracy_score
import numpy as np

start_time = time.time()
df=pds.read_csv('Assignment1.csv')

th=11

X_Training=[] #for input vals of x: 1 through 10

Y_Training=df['y'].tolist() #list of outputs
for i in range(len(df)):
    placeholder=[]
    for col in df:
        placeholder.append(df[col][i])
    
    X_Training.append(placeholder[:10]) #all input vals
  
def MCPNeuron(x,th): #MCP Neuron taking a length m input and then giving ans as per th value
    y_val_func=[] #collect vals from the MP Neuron function
    for row in x:
        if(sum(row)>=th): #aggregate of input vals
            y_val_func.append(1)
        else:
            y_val_func.append(0)
    
    return accuracy_score(Y_Training,y_val_func) #comparision for each th value

accu_max=[] #containing scores

for th in range(th):
    accu_max.append(MCPNeuron(X_Training,th))
    print("Threshold Value:")
    print(th)
    print("Accuracy Score")
    print(MCPNeuron(X_Training,th))
#print(MCPNeuron(X_Training,t))#last value for accuracy
#print(MCPNeuron)
pt.gcf().canvas.set_window_title('Graph')    
X_Axis=[x1 for x1 in range(0,th+1)]
pt.xticks(np.arange(0, 12, 1))
pt.plot(X_Axis,accu_max, color = 'red')
pt.xlabel('Threshold Value')
pt.ylabel('Accuracy')
pt.show()
print("--- %s seconds taken---" % (time.time() - start_time))





    
