Description of contents

Save all the following in the same directory

Data:
	-glove.bin--Baseline GloVe word vectors from Pennington et al
	
	-crawl-300d-2M-subword.bin--FastText word vectors 
	
	-stop_words.csv--List of stop words to be read in for text cleaning
	
	-Task Statements.xlsx--O*NET task descriptions
	
	-occupation_transitions_public_data_set.dta--Occupation to occupation worker transitions from "Employer Concentration and Outside Options" by Schubert, Stansbury, and Taska. Need to unzip the file. (obtained from Gregor Schubert's website: https://sites.google.com/view/gregorschubert) 

Scripts:
	-ONETOccupationSimilarities.py--Python script that cleans and prepares tasks statements, creates numerical representations of documents, and creates occupation-by-occupation task similarities. Outputs them to Stata datasets. 

	-SimilarityOccupationFlows.do--Reads in output from ONETOccupationSimilarities.py and uses it to predict worker occupation-to-occupation flows. Prints tables 3 and 4 of paper. Requires a few add-on Stata packages. See comment at top of script if not already installed
	
Run ONETOccupationSimilarities.py and SimilarityOccupationFlows.do sequentially 

Notes: 
	-ONETOccupationSimilarities.py was created using Python 3.9.12, but should work with Python >= 3.6.0. Gensim version 4.1.2 was used (see requirements.txt), but script should work with Gensim >= 4.0.0 
