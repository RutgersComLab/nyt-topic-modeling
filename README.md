# Topic modeling for NYT articles

## Overview
In this project, we try to identify how trends have changed for 'automation' across decades, by analyzing [The New York Times][nyt] articles from 1950 to 2021. We also try to identify other topics that managed to consistently get media attention over the decades. We use a variety of approaches and techniques for data cleaning and standardization, and also use packages such as [Gensim][gensim], [spaCy][spacy], [NLTK][nltk], [BERTopic][bertopic] etc. to complete the analysis.

## Problem Statements
The objective can be classified into two:
<ol>
	<li>Understand the trend around 'automation' using co-occurence of words.</li>
	<li>Understand how other topics have changed across decades using topic modeling.</li>
</ol>

## Implementation
The implemntation can broady be classified into the following stages:
<ol>
	<li>Creating the raw layer from source.</li>
	<li>Cleaning the data.</li>
	<li>Transforming the data</li>
	<li>Analyze the data</li>
	<ul>
		<li>Trend for Automation</li>
		<li>Trend for Other Topics</li>
	</ul>
</ol>
We discuss each of this in the following sections.

### Raw Layer
The source of data is from the [The New York Times][nyt] website. All the data for the raw layer reside as text files. We use [ProQuest][proquest] to get the news articles as text files. We have images of articles for the years 1950 to 1980, and text transcripts for the articles written after 1980. [ProQuest][proquest] uses an image to text tool to retrieve the articles from the images and return a text file. [ProQuest][proquest] also genereates a [metadata file](data/metadata/2020.csv) for the articles holding information like Publication info, Publication year, Publication subject, Abstract, Links, Location, People, Place of publication, etc. Each row of the metadata file holds information about one article. This file can be used to categorize the text files as needed. Sometimes, the image to text fails, leading to a mismatch in the number of rows in the metadata file and the actual number of files. An example of a text file generated by [ProQuest][proquest] for articles before 1980 can be [seen here](data/raw/114925995_1.txt), and another example of an article written after 1980 can be [seen here](data/raw/ProQuestDocuments-2020-10-160.txt). The folder structure for the raw layer is shown below:

![Folder structure of the Raw layer](screenshots/raw.png?raw=true "Folder structure of the Raw layer")

### Cleansed Layer
The first step involved in data cleaning is to categorize the articles based on the decade it was written. There are a few approaches taken here:
<ol>
	<li>1950-1959: No cleaning was required on the articles written during this decade.</li>
	<li>1960-1979: We use the metadata file generated by ProQuest to identify the files which falls into the 1960-1969 and 1970-1979 decades.</li>
	<li>1980-2020: Things get a bit more complicated here as the text files generated by ProQuest holds not only the content of the article, but also other metadata. Since the text files follow a particular structure, we use the rules of the structure to our advantage. By identifying the beginnnng and ending of the 'full text' section, we are able to extract just the articles from these files. We also apply the same logic to extract the 'publication year' from the text files, helping us categorize the article based on the decade. We also check if there are any outliers using the same logic.</li>
	<li>2010-2029: We use the rules used for 1980-2020 to identlify the 'publication year' from the text files, and categorize the article based on the decade. We also check if there are any outliers using the same logic.</li>
	<li>1950-1959: We encountered some outliers while processing 2010-2019. Upon further inspection, these turned out to be files from 2009. We manually move them to the appropriate folder.</li>
</ol>

The code for this layer is available [here](code/automation/Data%20Cleaning%20and%20Transformation.ipynb). The folder structure for the cleaned layer is shown below:

![Folder structure of the Cleaned layer](screenshots/cleaned.png?raw=true "Folder structure of the Cleaned layer")

### Transformed Layer
This is the final layer before starting the analysis. We apply a variety of rules in this layer to prepare the data for subsequent layers. First, we remove stopwords, prepositions, punctuations, pronouns and common words from all the articles. The folder structure after this has been applied on the cleaned layer looks like:

![Folder structure of the Transformed layer](screenshots/transformed.png?raw=true "Folder structure of the Transformed layer")

Next, we create unigrams, bigrams and trigrams for each article. We also get the count of each unigram, bigram and trigram generated per article, and format the results into a JSON of format:
```
[
    ('id', identifier),
    ('outputFormat', ['unigram', 'bigram', 'trigram', 'fullText']),
    ('wordCount', wordCount),
    ('fullText', fullText),
    ('unigramCount', unigramCount), 
    ('bigramCount', bigramCount), 
    ('trigramCount', trigramCount)
]
```
Finally, we save the generated JSONs as JSONL files, and do some validation to make sure we have not lost any articles between the cleaned and transformed layers. The code for this layer is available [here](code/automation/Data%20Cleaning%20and%20Transformation.ipynb).

![JSONL files](screenshots/jsonl.png?raw=true "JSONL files")

### Analysis
The analysis can be split into two sections depending on our task:
#### Trend for Automation
To understand how the trends have changed for 'automation' across decades, we look at the frequency of occurence fro the word automation in the articles through 3 different persepetives:
<ol>
	<li>count/id - Aims to answer how many times the word automation occured as a unigram, bigram or trigram in the corpus for each decade, on a file level.</li>
	<li>count/word - Aims to answer how many times a bigram or trigram was present in the corpus for each decade, on a word level.</li>
	<li>count/lemmatized - Aims to answer how many times a bigram or trigram was present in the corpus for each decade, on a word level, after lemmatization</li>
</ol>
We build a visualization on top of count/lemmatized to get a better sense of the distrbution of the top 50 words in the corpus across the decades.

#### Trend for Other Topics
This project approaches the problem using 4 different techniques:
Decade wise topic modeling using LDA
Decade wise topic modeling using BERTopic
Dynamic Topic Modeling (DTM) using LDA Sequence Model
Dynamic Topic Modeling (DTM) using BERTopic - DTM

## Output

## References
LDA
LDASeq
BERTopic
BERTopic-DTM
NLTK
Gensim
spaCy
Plotly
Pandas
Numpy
Haystack
scikit-learn
pyLDAvis
The New York Times
ProQuest

[lda]:https://radimrehurek.com/gensim/models/ldamodel.html
[lda_Seq]:https://radimrehurek.com/gensim/models/ldaseqmodel.html
[bertopic]:https://github.com/MaartenGr/BERTopic/
[bertopic_dtm]:https://github.com/MaartenGr/BERTopic/#dynamic-topic-modeling
[nltk]:https://www.nltk.org/
[gensim]:https://radimrehurek.com/gensim/
[spacy]:https://spacy.io/
[plotly]:https://plotly.com/
[pandas]:https://pandas.pydata.org/
[numpy]:https://numpy.org/
[haystack]:https://haystack.deepset.ai/overview/intro
[scikit_learn]:https://scikit-learn.org/stable/index.html
[pyldavis]:https://github.com/bmabey/pyLDAvis
[nyt]:https://www.nytimes.com/
[proquest]:https://www.proquest.com/