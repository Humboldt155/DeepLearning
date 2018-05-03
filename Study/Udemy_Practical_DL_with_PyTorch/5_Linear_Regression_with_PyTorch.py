#%% Импортируем библиотеки

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#%% Объявляем x и y

x_train = [x for x in range(11)]
y_train = [2 * x + 1 for x in x_train]

x_train = np.array(x_train, dtype=np.float32).reshape(-1, 1)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)



#%% Create class

# функция y = 2 * x + 1

# Объявление класса делается каждый раз, когда создается модель
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):

        # python superfunction, копирует весь функционал класса
        # добавляется, чтобы получить весь фнукционал nn.Module
        super(LinearRegressionModel, self).__init__()

        # input_dim - x, output_dim - y
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


#%% Instantiate model ( Объявляем модель )

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

#%% Instantiate loss class ( объявляем функцию потерь)

learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Тренировка модели

""" Процесс тренировки модели:
        
        Один цикл называется Эпохой.
        Эпоха - это значит пройти через все обучающие данные один раз
        
        1. Конвертируем input в переменные
        2. Очистить gradient buffets
        3. Получить выходный данные
        4. Получить потери
        5. Получить параметры w.r.t. градиента
        6. Обновить параметры используя градиент
            parameters = parameters - learning_rate * parameters_gradients
        7. ПОВТОРИТЬ
"""

epochs = 100

for epoch in range(epochs):
    epoch += 1

    # convert numpy variables to tensors
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)


    # clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # forward to get outputs
    outputs = model(inputs)

    # calculate loss
    loss = criterion(outputs, labels)

    # getting gradients w.r.t. parameters
    loss.backward()

    # updating parameters
    optimizer.step()

    print('epoch: {}, loss: {}'.format(epoch, loss.data[0]))


#%% Check

y_pred = model(torch.from_numpy(x_train)).data.numpy()

#%% Save model

save_model = True

if save_model is True:
    # Saves only parameters
    # a and b
    torch.save(model.state_dict(), 'lrm.pkl')

#%% Load model

load_model = True

if load_model is True:
    model.load_state_dict(torch.load('lrm.pkl'))