Description of contents

Save all the following files in the same directory and run from that directoryu

---------------------------------------
For reproducing Tables 3 and 4:
---------------------------------------
Data:
	-glove.bin--Baseline GloVe word vectors from Pennington et al
	-crawl-300d-2M-subword.bin--FastText word vectors 
	-stop_words.csv--List of stop words to be read in for text cleaning
	-Task Statements.xlsx--O*NET task descriptions
	-occupation_transitions_public_data_set.dta--Occupation to occupation worker transitions
         from "Employer Concentration and Outside Options" by Schubert, Stansbury, and Taska
         (obtained from Gregor Schubert's website: https://sites.google.com/view/gregorschubert) 


Scripts:
	-ONETOccupationSimilarities.py--Python script that cleans and prepares tasks statements, creates numerical representations of documents, and creates occupation-by-occupation task similarities. Outputs them to Stata datasets. 

	-SimilarityOccupationFlows.do--Reads in output from ONETOccupationSimilarities.py and uses it to predict worker occupation-to-occupation flows. Prints tables 3 and 4 of paper 
		Requires a few additional Stata packages. See comment at top of script if not already installed

Notes: 
	-ONETOccupationSimilarities.py was created using Python 3.9.12, but should work with Python >= 3.6.0
         Gensim version 4.1.2 was used (see requirements.txt), but script should work with Gensim >= 4.0.0 

--------------------------------------------
For reproducing Table 2
--------------------------------------------

Data:	
	-naics4_top100_patent_similarities.dta--Table containing top 100 year-2000 breakthrough patents 
	 for each NAICS code,including patent names and numbers and 4-digit NAICS titles

	-naics_descriptions.csv--Original NAICS descriptions that we scraped from the NAICS
	 manual website: https://www.census.gov/naics/?58967?yearbck=2012. We defined NAICS at the 
         4-digit level and combine the fields "naics6_text", "naics5_text", "naics4_text" (as available)
         along with "naics4_title" to create one document for each 4-digit naics code.  Included for 
	 completion, though not read in directly by any replication scripts. 

Scripts:
	 -NAICSPatentSimilarityTable.do--Run to replicate each panel of Table 2. 


NOTES:

As the data are too large, we don't include the raw patent texts. We instead provide
a dataset with the top 100 patents for each NAICS code, which allows one to replicate Table 2
and to examine additional examples. 

Data for Figure 1 and Table 1 come from "Technology-Skill Complementarity and Labor Displacement:
Evidence from Linking Patents Occupations", 2021 by Kogan, Papanikolaou, Schmidt, and Seegmiller, 
and are not included here. 




	 

