#%% Импортируем библиотеки

import torch
import torch.nn as nn
from torch.autograd import Variable

#%%



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

learning_rate = 0.01

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


