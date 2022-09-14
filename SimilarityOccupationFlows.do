//Computes Tables 3 and 4 in "Measuring Document Similarity with Weighted Averages of Word Embeddings"
// by Seegmiller, Papanikolaou, and Schmidt 

/*
//To install packages if not done already
ssc install ftools //Necessary for reghdfe 
ssc install reghdfe 
ssc install fastreshape
ssc install gtools //To get gquantiles command for computing fast percentiles 
ssc install estout 
*/

//Reading in similarities from tf-idf weighted avg of word embeddings (GloVe)
use PairwiseOccSimilarityONET.dta, clear 
rename soc soc1 
fastreshape long occ_code, i(soc1) j(soc2)
rename occ_code rho //rho will label the raw similarity score 
drop if soc1 == soc2 //Only keep off-diagonal entries 
//Because word embeddings are dense vectors the sim scores are not sparse. 
//Transformation of raw score from Kogan, Papanikolaou, Schmidt, and Seegmiller
gquantiles rho100 = rho, nq(100) xtile 
egen max_rho = max(rho)
sum rho if rho100 == 80 
gen p80_rho = r(mean)
gen rho_tilde = (rho-p80_rho)*(rho>=p80_rho)/(max_rho-p80_rho) 
drop max_rho p80_rho 
save PairwiseOccSimilarityONET_long.dta, replace 

//Reading in similarities from tf-idf weighted avg of word embeddings (FastText)
use PairwiseOccSimilarityONET_FastText.dta, clear 
rename soc soc1 
fastreshape long occ_code, i(soc1) j(soc2)
rename occ_code rho_ft //rho will label the raw similarity score 
drop if soc1 == soc2 //Only keep off-diagonal entries 
//Because word embeddings are dense vectors the sim scores are not sparse. 
//Transformation of raw score from Kogan, Papanikolaou, Schmidt, and Seegmiller
gquantiles rho_ft100 = rho_ft, nq(100) xtile 
egen max_rho = max(rho_ft)
sum rho_ft if rho_ft100 == 80 
gen p80_rho = r(mean)
gen rho_ft_tilde = (rho_ft-p80_rho)*(rho_ft>=p80_rho)/(max_rho-p80_rho) 
drop max_rho p80_rho 
save PairwiseOccSimilarityONET_long_ft.dta, replace 

//Reading in similarities from tf-idf bag of words  
use PairwiseOccSimilarityONET_TfidfBow.dta, clear 
rename soc soc1 
fastreshape long occ_code, i(soc1) j(soc2)
rename occ_code rho_bow //rho will label the raw similarity score 
drop if soc1 == soc2 //Only keep off-diagonal entries 
//Try same transformation from Kogan, Papanikolaou, Schmidt, and Seegmiller
gquantiles rho_bow100 = rho_bow, nq(100) xtile 
egen max_rho_bow = max(rho_bow)
sum rho_bow if rho_bow100 == 80 
gen p80_rho_bow = r(mean)
gen rho_bow_tilde = (rho_bow-p80_rho_bow)*(rho_bow>=p80_rho_bow)/(max_rho_bow-p80_rho_bow) 
drop max_rho_bow p80_rho_bow 
save PairwiseOccSimilarityONET_long_TfidfBow.dta, replace 

//Reading in similarities from LDA with 100 Topics
use PairwiseOccSimilarityONET_LDA.dta, clear 
rename soc soc1 
fastreshape long occ_code, i(soc1) j(soc2)
rename occ_code rho_lda //rho will label the raw similarity score 
drop if soc1 == soc2 //Only keep off-diagonal entries 
save PairwiseOccSimilarityONET_long_lda.dta, replace 

use occupation_transitions_public_data_set.dta, clear 
replace soc1 = subinstr(soc1,"-","",3)
replace soc2 = subinstr(soc2,"-","",3)
destring soc1 soc2, replace force 

merge 1:1 soc1 soc2 using PairwiseOccSimilarityONET_long.dta, keep(match master) nogen 
merge 1:1 soc1 soc2 using PairwiseOccSimilarityONET_long_ft.dta, keep(match master) nogen 
merge 1:1 soc1 soc2 using PairwiseOccSimilarityONET_long_TfidfBow.dta, keep(match master) nogen 
merge 1:1 soc1 soc2 using PairwiseOccSimilarityONET_long_lda.dta, keep(match master) nogen 

gen logshare = log(transition_share)

label variable rho "Raw Similarity, $\rho_{i,j}$"
label variable rho_tilde "Adjusted Similarity, $\tilde{\rho}_{i,j}$"
label variable rho100 "Similarity Pctile, $\rho_{i,j}^{pctile}$"
label variable rho_ft "Raw Similarity, $\rho_{i,j}^{FastText}$"
label variable rho_ft_tilde "Adjusted Similarity, $\tilde{\rho}_{i,j}^{FastText}$"
label variable rho_ft100 "Similarity Pctile, $\rho_{i,j}^{pctile,FastText}$"
label variable rho_bow "Raw Similarity, $\rho_{i,j}^{BOW}$"
label variable rho_bow_tilde "Adjusted Similarity, $\tilde{\rho}_{i,j}^{BOW}$"
label variable rho_bow100 "Similarity Pctile, $\rho_{i,j}^{pctile,BOW}$"
label variable rho_lda "Raw Similarity, $\rho_{i,j}^{LDA}$"

est clear
//Baseline GloVe
eststo a1: reghdfe logshare rho [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo a2: reghdfe logshare rho [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo a3: reghdfe logshare rho_tilde [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo a4: reghdfe logshare rho_tilde [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo a5: reghdfe logshare rho100 [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo a6: reghdfe logshare rho100 [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"

//Tf-Idf Bag of Words
eststo b1: reghdfe logshare rho_bow [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo b2: reghdfe logshare rho_bow [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo b3: reghdfe logshare rho_bow_tilde [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo b4: reghdfe logshare rho_bow_tilde [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo b5: reghdfe logshare rho_bow100 [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo b6: reghdfe logshare rho_bow100 [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"

//Alternate FastText
eststo c1: reghdfe logshare rho_ft [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo c2: reghdfe logshare rho_ft [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo c3: reghdfe logshare rho_ft_tilde [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo c4: reghdfe logshare rho_ft_tilde [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo c5: reghdfe logshare rho_ft100 [aw=total], noab cluster(soc1 soc2)
estadd local occfe ""
eststo c6: reghdfe logshare rho_ft100 [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"

//Compare w/ LDA 
eststo d1: reghdfe logshare rho [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo d2: reghdfe logshare rho_bow [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo d3: reghdfe logshare rho_lda [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo d4: reghdfe logshare rho rho_bow rho_lda [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"
eststo d5: reghdfe logshare rho_ft rho_bow rho_lda [aw=total], ab(soc1 soc2) cluster(soc1 soc2)
estadd local occfe "X"

//Table 3 Panel A
esttab a*, keep(rho rho_tilde rho100) s(occfe N r2_within,label("Occ FE" "N" "$\text{R}^2$ (Within)")) label 	
//Table 3 Panel B 
esttab b* , keep(rho_bow rho_bow_tilde rho_bow100) s(occfe N r2_within,label("Occ FE" "N" "$\text{R}^2$ (Within)")) label 

//Table 4 Panel A 
esttab c* , keep(rho_ft rho_ft_tilde rho_ft100) s(occfe N r2_within,label("Occ FE" "N" "$\text{R}^2$ (Within)")) label 
//Table 4 Panel B 
esttab d* , keep(rho rho_ft rho_bow rho_lda) order(rho rho_ft rho_bow rho_lda) s(occfe N r2_within,label("Occ FE" "N" "$\text{R}^2$ (Within)")) label 
