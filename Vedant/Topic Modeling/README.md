# Topic Modeling

## Set up

* Create New Conda enviornment
	- conda create --name enviornment_name python=3.7 (We use python 3.7 because it is the enviornment in which the topic model was trained.)
	- conda activate enviornment_name

* Install all the packages in package.txt file
	- pip install package.txt

## Folder Structure

. Topic Modeling Code
	|____ BERT
		|___BERTOPIC.ipynb
		|___BERTDTM_FINAL.ipynb
		|___Results
		|___Model Files
			|___ BERT_DTM
			|___ BERT
	|____data
	|____LDA

Inside Topic Modeling Code folder their are 3 folders named BERT, data and LDA. 

### BERT Folder
The BERT Folder contains all the code for BERTopic Models predictions, the models were trained by Jacob Celestine, I used the trained models to make predictions and graphs for the data. 

#### BERTopic.ipynb
BERTopic.ipynb contains all the graphs generated for BERT model trained on decadely data. To run the file simply type the command:

	- jupyter notebook BERTopic.ipynb

#### BERTDTM_FINAL.ipynb
BERTDTM_FINAL.ipynb contains all the graphs generated for BERT Dynamic Topic Model. To run the file:

	- jupyter notebook BERTDTM_FINAL.ipynb

#### Model Files
This folder contains the pickle file for the trained models. The BERT folder contains the pickle files for BERT Topic Model trained on decade wise data so it has folders named 1950-1959 etc. The folder means the topic model was trained on articles between that time and all the relevant pickle files are stored in that folder. BERT_DTM folder contains the pickle files that are for BERT Dynamic topic model and if one needs to generate new graphs for these models one can simply load the models as done in both the notebooks and run the command to generate the graphs.

#### Results
This folder contains all the results of Topic modeling.

### data
This folder contains all the data files as the name suggests.

### LDA
This folder contains the code for LDA models.

