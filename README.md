# Movie_Recommendation
This project implements a hybrid neural network architecture to predict a user's next preferred movie genre using the MovieLens dataset. By combining spatial feature extraction with sequential modeling, the system achieves 92% accuracy in top 10 genre forecasting.

## Architecture
The model utilizes a stacked approach to handle both the complexity of user preferences and the chronological nature of viewing habits:
* Stacked Autoencoder (SAE): Used to compress high-dimensional user-item interactions into a dense latent representation, capturing underlying "taste" profiles.
* LSTM (Long Short-Term Memory): Processes these latent representations over time to identify patterns in how a user's genre interests evolve.

## EDA 
Before modeling, initial EDA was done using python in order to understand the distribution of data. Visualization was done to look into movie genere distribution, age distribution by gender, and job occupation by gender. 

## Source Files and Folders
* movie_recommendation.py
* gender_by_age_plot.png
* ml-100k
* ml-1m
* ml-latest-small
