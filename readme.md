This repository is to provide the dataset to the article 'Improving model performance on patient stratification by integrating multiscale genomic features'. Please note that this code is ONLY provided for academic use.

File list:
BRCAmiRNAseq_matrix_out_cv.arff:		Training dataset of micro RNA in ARFF format. Could be used for model building and cross validation.
BRCAmiRNAseq_matrix_out_ind.arff:		Test dataset of micro RNA in ARFF format. Could be used for model independent test.
BRCAmRNA_FPKM_matrix_out_cv.arff:	Training dataset of mRNA in ARFF format. Could be used for model building and cross validation.
BRCAmRNA_FPKM_matrix_out_ind.arff:	Test dataset of mRNA in ARFF format. Could be used for model independent test.
usingshap_RF.py:						The code for generating the shrinked dataset by using model shap (https://github.com/slundberg/shap).

Python model requirement:
1) shap： https://github.com/slundberg/shap
2) arff
3) scipy
4) pandas
5) numpy
6) sklearn
All the models could be istalled by using pip, and models 3) - 5) are already included in Anaconda.

A example of using the code for a training dataset and a test dataset:
`python usingshap_RF.py BRCAmRNA_FPKM_matrix_out_cv.arff BRCAmRNA_FPKM_matrix_out_ind.arff`

A example of using the code for two training datasets and two test datasets:
`python usingshap_RF.py BRCAmiRNAseq_matrix_out_cv.arff BRCAmiRNAseq_matrix_out_ind.arff BRCAmRNA_FPKM_matrix_out_cv.arff BRCAmRNA_FPKM_matrix_out_ind.arff`

Both examples will yield 3 files:
    selectedDataSetTrain.arff:  The shrinked training dataset in ARFF format.
    selectedDataSetTest.arff:   The shrinked test dataset in ARFF format.
    selectedFeaName.txt:        The selected features.
    
Please rename the file after generated as needed.