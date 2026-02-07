# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

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
### Name: V KAMALESH VIJAYAKUMAR
### Register Number:212224110028
```python
#creating model class
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

#Function to train model
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

<img width="160" height="269" alt="image" src="https://github.com/user-attachments/assets/9b5c05df-0a9d-4cac-87f1-fe35d774b0e2" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="724" height="582" alt="image" src="https://github.com/user-attachments/assets/f56007d7-a823-42a6-aa99-1637c7ad308b" />


### New Sample Data Prediction
<img width="339" height="34" alt="image" src="https://github.com/user-attachments/assets/f90961fc-25cc-4b63-87e8-5554db28f9e2" />


## RESULT
Thus the neural network regression model is developed using the given dataset.
