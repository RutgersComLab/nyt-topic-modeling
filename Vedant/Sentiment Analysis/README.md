# Sentiment Analysis

# Set up

* Create New Conda enviornment
	- conda create --name enviornment_name python=3.7 (We use python 3.7 because it is the enviornment in which the topic model was trained.)
	- conda activate enviornment_name

* Install all the packages in package.txt file
	- pip install package.txt

# Folder Structure

. Sentiment Analysis
	|___ Results
        |___ Sentiment Analysis Graphs
    |___ Code
        |___ Longformer
            |___ Code
                |___ Sentiment_Analysis_Longformer.py
                |___ Sentiment_Prediction_Longformer.py
                |___ Generate_Graphs.ipynb
                |___ Map_articles_to_year.ipynb
                |___ Data
                    |___ pq_metadata.csv
                    |___ sentiment_store.pickle
                    |___ final_map.pkl
                    |___ final_map_year.pkl
            |___ Model Files
                |___ longformer__all__fine__e4.pickle
        |___BERT
            |___ Code
                |___ Sentiment_Analysis.py
                |___ Sentiment_Prediction.py
                |___ Map_articles_to_year.ipynb
                |___ Map_articles_to_sentiment.ipynb
                |___ Generate_graphs.ipynb
                |___ Data
                    |___ final_map_year.pkl
                    |___ final_map.pkl
                    |___ pq_metadata.csv
                    |___ sentiment_store.pickle

Inside Sentiment Analysis folder there there are two folder namely Results and Code. 

## Results
This folder contains all the results of Sentiment Analysis.

## Code

Inside Code their are two folders Longformer and BERT.

### Longformer Folder

#### Model Files

This folder contains the model pickle i.e. the trained Longformer model weights which can be used to make predictions and generate graphs.

#### Code

This folder contains all the code. Use of each file and how to run the file is explained below:

##### Sentiment_Analysis_Longformer.py

This file is used to train the longformer model on the SST-5 dataset. Longformer is a very big model and so to train it I trained the model on ilab servers using GPU it can be done as follows:

    - ssh netid@ilab1.cs.rutgers.edu (SSH into the host if asked for password enter you cs.rutgers.edu password)
    - Open a terminal in the host using screen (It prevents the university people from killing your process)
    - Srun -G1 python3 Sentiment_Analysis_Longformer.py (This trains the model on the GPU)
    
    * This might throw error that some packages are not intalled don't worry just install the latest packages for longformer and the code will work.

##### Sentiment_Prediction_Longformer.py

This file is used to make Sentiment predictions on the nyt dataset. Similar to the previous approach run this file.

##### Generate_Graphs.ipynb

This file generates all the graphs for longformer just load the sentiment_store.pickle file and run all the cells.

# Code Flow for Longformer and BERT

The proper code flow is as follows:

1. Run the Map_articles_to_sentiment.ipynb to generate final_map pickle.
2. Run Map_articles_to_year.ipynb to generate final_map_year pickle which is used in the training and prediction files.
3. Train the model using Sentiment_Analysis.py file
4. Make predictions using Sentiment_Predictions.py file
5. Generate graphs using Generate_Graphs.ipynb file.
