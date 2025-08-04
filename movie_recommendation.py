# Importing the libraries

# Data imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

# Modelling imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Metric imports
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Visualization imports
from plotnine import *
import matplotlib.pyplot as plt

## 1. Creating Visualizations

# Importing the dataset

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, names=['movie_id', 'title', 'genres'], engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Encoding values for visual analysis

users_age_map = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-49',
    45: '45-49',
    50: '50-55',
    56: '56+'
    }

users_occupation_map = {
    0: 'other or not specified',
	1: 'academic/educator',
	2: 'artist',
	3: 'clerical/admin',
	4: 'college/grad student',
	5: 'customer service',
	6: 'doctor/health care',
	7: 'executive/managerial',
	8: 'farmer',
	9: 'homemaker',
	10: 'K-12 student',
	11: 'lawyer',
	12: 'programmer',
	13: 'retired',
	14: 'sales/marketing',
	15: 'scientist',
	16: 'self-employed',
	17: 'technician/engineer',
	18: 'tradesman/craftsman',
	19: 'unemployed',
	20: 'writer'
    }

users[2] = users[2].replace(users_age_map)
users[3] = users[3].replace(users_occupation_map)
users.rename(columns = {1: 'gender', 2: 'age_group', 3: 'occupation'}, inplace = True)
users_filtered_age = users[users['age_group'] != 'Under 18']
users_filtered_employed = users[~users['occupation'].isin(['unemployed', 'other or not specified'])]

movies_genres_separated = movies['genres'].str.split('|').explode()
movies_with_separated_genres = pd.concat([movies[['movie_id', 'title']], movies_genres_separated], axis=1)


# Visualizing the data

ggplot(users_filtered_age, aes(x = 'gender', fill = 'gender')) + geom_bar() + geom_text(aes(label = '..count..'), stat = 'count', va = 'bottom', size = 8, nudge_y = -120)+ facet_wrap('~age_group') + theme_bw(base_size = 9) + xlab("Gender")

ggplot(users_filtered_employed, aes(x = 'gender', fill = 'gender')) + geom_bar() + geom_text(aes(label = '..count..'), stat = 'count', va = 'bottom', size = 5, nudge_y = -50)+ facet_wrap('~occupation') + theme_bw(base_size = 9) + xlab("Gender")

ggplot(movies_with_separated_genres, aes(x='genres', fill = 'genres')) + geom_bar() + xlab("Genres") + ylab("Count") + theme(axis_text_x=element_text(rotation=45, hjust=1))

## 2. Creating the stacked autoencoder

# Preparing the training and testing set for the autoencoder

training_set = pd.read_csv('ml-100k/u3.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u3.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of movies and users

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


# Converting training and test set into vectorized form for autoencoder

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append((list(ratings)))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
    
# Converting this data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the stacked autoencoder

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, nb_movies)
        self.activation = nn.Sigmoid()
        
        # Added dropouts to slightly help with overfitting
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    
    def encoder(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        return x
    
    def decoder(self, x):
        x = self.activation(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
# optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
optimizer = optim.Adam(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epochs = 50
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0 # Make sure these values wont count for calculating error
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + 'Loss: ' + str(train_loss/s))
    
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(id_user):
    input = Variable(training_set[id_user]).unsqueeze(0) # Training set is the movies user rated before. We use this data to compare the data that the training set did not have. Compare predicted data to the real ratings. 
    target = Variable(test_set[id_user]).unsqueeze(0) # Target contains real values we want to compare to
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0 # Make sure these values wont count for calculating error
        loss = criterion(output,target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('Test loss: ' + str(test_loss/s))

## 3. Creating the RNN(LSTM) for movie genre prediction

# Prepping the data for the RNN

# Data Encoding

user_embedded_data = []

for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    with torch.no_grad():
        output = sae.encoder(input).squeeze(0)
        user_embedded_data.append(output.numpy())
        
sc = MinMaxScaler()
user_embedded_data_df = pd.DataFrame(user_embedded_data)
user_embedded_data_df['user_id'] = range(1, nb_users + 1)
columns_to_scale = user_embedded_data_df.columns.drop('user_id')
scaled_data = sc.fit_transform(user_embedded_data_df[columns_to_scale])
user_embedded_data_df[columns_to_scale] = scaled_data

training_set = pd.read_csv('ml-100k/u3.base', delimiter = '\t', names = ['user_id', 'movie_id', 'rating', 'timestamp'])
training_set = training_set.sort_values(by = ['user_id', 'timestamp'])          

mlb = MultiLabelBinarizer()  
le = LabelEncoder()
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
genre_encoded = pd.DataFrame(mlb.fit_transform(movies['genres']), columns= mlb.classes_, index = movies.index)
movies = pd.concat([movies[['movie_id']], genre_encoded], axis = 1)
movies['movie_id'] = le.fit_transform(movies['movie_id'])
            
training_set = pd.merge(training_set, movies, on = 'movie_id', how = 'inner')
training_set['timestamp'] = pd.to_datetime(training_set['timestamp'], unit = 's')
training_set = pd.merge(training_set, user_embedded_data_df, on = 'user_id', how = 'inner')
training_set['movie_id'] = le.fit_transform(training_set['movie_id'])

genres = mlb.classes_.tolist() # columns for genres
embedded = user_embedded_data_df.columns.difference(['user_id']).tolist() # columns for embedded user data
features = genres + embedded

X_train = []
Y_train = []
sequence_length = 5

unique_users = training_set.groupby('user_id')
min_user = unique_users.size().min()

# Filling X_train and Y_train
for user_id, group in unique_users:
    for i in range(sequence_length, len(group)):
        current_item = group.iloc[i - sequence_length:i]
        next_item = group.iloc[i]
        
        X_train.append(current_item[features])
        Y_train.append(next_item[genres])

X_train, Y_train = np.array(X_train, dtype=np.float32), np.array(Y_train, dtype=np.float32)

# Initializing the RNN

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 82)))
regressor.add(Dropout(0.2)) 

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 60, return_sequences = False))
regressor.add(Dropout(0.2)) 

# Adding the output layer
regressor.add(Dense(units = len(genres), activation = 'sigmoid'))

# Compiling the RNN
top_10 = TopKCategoricalAccuracy(k = 10) 
adam_learning_rate = Adam(learning_rate = 0.1)
regressor.compile(optimizer = adam_learning_rate, loss = 'binary_crossentropy', metrics = ['accuracy', top_10])

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# Testing the RNN 

user_embedded_data_y = []

for id_user in range(nb_users):
    input = Variable(test_set[id_user]).unsqueeze(0)
    target = input.clone()
    with torch.no_grad():
        output = sae.encoder(input).squeeze(0)
        user_embedded_data_y.append(output.numpy())
        
user_embedded_data_df_y = pd.DataFrame(user_embedded_data_y)
user_embedded_data_df_y['user_id'] = range(1, nb_users + 1)

sc = MinMaxScaler()
user_embedded_data_df_y = pd.DataFrame(user_embedded_data_y)
user_embedded_data_df_y['user_id'] = range(1, nb_users + 1)
scaled_data = sc.fit_transform(user_embedded_data_df_y[columns_to_scale])
user_embedded_data_df_y[columns_to_scale] = scaled_data

test_set = pd.read_csv('ml-100k/u3.test', delimiter = '\t', names = ['user_id', 'movie_id', 'rating', 'timestamp'])
test_set = test_set.sort_values(by = ['user_id', 'timestamp'])
test_set = pd.merge(test_set, movies, on = 'movie_id', how = 'inner')
test_set['timestamp'] = pd.to_datetime(test_set['timestamp'], unit = 's')
test_set = pd.merge(test_set, user_embedded_data_df_y, on = 'user_id', how = 'inner')
test_set['movie_id'] = le.fit_transform(test_set['movie_id'])

X_test = []
Y_test = []

unique_users_y = test_set.groupby('user_id')

for user_id, group in unique_users_y:
    if len(group) >= sequence_length + 1: 
        for i in range(sequence_length, len(group)):
           current_item = group.iloc[i - sequence_length:i]
           next_item = group.iloc[i]
            
           X_test.append(current_item[features])
           Y_test.append(next_item[genres])
       
X_test, Y_test = np.array(X_test, dtype=np.float32), np.array(Y_test, dtype=np.float32)
        
regressor.evaluate(X_test, Y_test)

