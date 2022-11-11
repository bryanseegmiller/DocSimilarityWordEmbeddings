//------------------------------------------------------------------------------
//Output panels A through D of Table 2 from 
//"Measuring document similarity with weighted averages of word embeddings"
//by Seegmiller, Papanikolaou, and Schmidt (2022)
//------------------------------------------------------------------------------
//Note: Before running, should do "ssc install texsave"

//Dataset includes top 100 most similar patents to each 4-digit NAICS code, 
//along with textual similarity scores, patent titles and naics titles. 
use naics4_top100_patent_similarities.dta, clear 
egen rank = rank(-rho), by(naics)
keep if rank <= 5 //Top 5 patents per industry 
sort naics rank 
drop rank  

//Relabeling variables for purposes of outputting tables to Latex
label variable patnum "\textbf{Patent Number}"
label variable patent_name "\textbf{Patent Title}"
label variable rho "$\rho_{i,j}$"

//Panel A: Oil and Gas Extraction (NAICS Code 2111)
texsave rho patnum patent_name if naics == 2111 using ///
		"panelA_naics2111_top_patents.tex", ///
		frag varlabels replace location(h) size(footnotesize)  align(cC)  ///
		rowsep(.18cm) headlines("Oil and Gas Extraction (NAICS Code 2111)") 

//Panel B: Aerospace Product and Parts Manufacturing (NAICS Code 3364)
texsave rho patnum patent_name if naics == 3364 using ///
		"panelB_naics3364_top_patents.tex", ///
		frag varlabels replace location(h) size(footnotesize)  align(cC)  ///
		rowsep(.18cm) headlines("Aerospace Product and Parts Manufacturing (NAICS Code 3364)") 

//Panel C: Software Publishers (NAICS Code 5112)		  
texsave rho patnum patent_name if naics == 5112 using ///
		"panelC_naics5112_top_patents.tex", ///
		frag varlabels replace location(h) size(footnotesize)  align(cC)  ///
		rowsep(.18cm) headlines("Software Publishers (NAICS Code 5112)") 
		  
//Panel D: Restaurants and Other Eating Places (NAICS Code 7225)
texsave rho patnum patent_name if naics == 7225 using ///
		"panelD_naics7225_top_patents.tex", ///
		frag varlabels replace location(h) size(footnotesize)  align(cC)  ///
		rowsep(.18cm) headlines("Restaurants and Other Eating Places (NAICS Code 7225)")		