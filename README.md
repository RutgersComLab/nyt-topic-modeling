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
	<li>Identiyfing the data source.</li>
	PDF to text
	text directly from website
	<li>Cleaning the data.</li>
	<li>Transforming the data</li>
	<li>Analyze the data</li>
	<ul>
		<li>Trend for Automation</li>
		To understand how the trends have changed for 'automation' across decades, we look at the frequency of occurence fro the word automation in the articles through 3 different persepetives:
		<ol>
			<li>count/id - Aims to answer how many times the word automation occured as a unigram, bigram or trigram in the corpus for each decade, on a file level.</li>
			<li>count/word - Aims to answer how many times a bigram or trigram was present in the corpus for each decade, on a word level.</li>
			<li>count/lemmatized - Aims to answer how many times a bigram or trigram was present in the corpus for each decade, on a word level, after lemmatization</li>
		</ol>
		<li>Trend for Other Topics</li>
	</ul>
</ol>

### Source
PDF to text
text directly from website
### Data Cleaning
categorize years
extract text ad categorize on years
extract text
fix outliers
### Data Transformation
tokenize text and remove stopwords, prepositions, punctuations, pronouns and common words
create unigram, bigrams and trigrams for each decade corpus

### Analyzis
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