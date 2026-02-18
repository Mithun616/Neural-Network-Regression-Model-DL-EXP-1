# Developing a Neural Network Regression Model
## NAME: Mithun Kumar G
## REG NO:212224230160
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="930" height="643" alt="image" src="https://github.com/user-attachments/assets/eaebc717-17d1-4762-89a3-b584d2b05979" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:MITHUN KUMAR G
### Register Number: 212224230160
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def mithunkumar():
    print("Name: Mithun Kumar")
    print("Register Number: 212224230160")

dataset1 = pd.read_csv('MyMLData.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
        mithunkumar()
        print("Neural Network Regression Model Initialized")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_brain = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    mithunkumar()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    mithunkumar()
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()

mithunkumar()
print(f'Prediction for input 9: {prediction}')

```
## Dataset Information

<img width="142" height="372" alt="image" src="https://github.com/user-attachments/assets/c649d629-f957-4dfa-b161-784515b17fa0" />

## OUTPUT

<img width="426" height="308" alt="image" src="https://github.com/user-attachments/assets/dcd978b8-f9b9-4e7f-ac73-22a7d812d303" />

### Training Loss Vs Iteration Plot

<img width="574" height="454" alt="image" src="https://github.com/user-attachments/assets/9f6f1f25-2759-43ee-9311-fe751b3367a0" />


### New Sample Data Prediction

<img width="340" height="61" alt="image" src="https://github.com/user-attachments/assets/21028c89-e50c-4667-b27a-a96f473ab9ff" />

## RESULT

Successfully executed the code to develop a neural network regression model.

