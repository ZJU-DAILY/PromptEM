# Machamp

Machamp is a Benchmarking for the task of Generalized Entity Matching (GEM), which aims at performing entity matching between entries in structured, semi-structured, and unstructured format. 

## Task Description

* Rel-HETER: This task is for matching between structured tables with heterogeneous schema. The source is the Fodors-Zagats task from the [Deep Matcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) datasets.

* Semi-HOMO: This task is for matching between semi-structured tables with homogeneous schema. The source is the DBLP-Scholar task from the [Deep Matcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) datasets.

* Semi-HETER: This task is for matching between semi-structured tables with heterogeneous schema. The source is the set of 5 Book tasks from the [Magellan](https://sites.google.com/site/anhaidgroup/useful-stuff/data#TOC-The-784-Data-Sets-for-EM) project.

* Semi-Rel: This task is for matching between semi-structued and structured tables. The source is the set of 5 Movie tasks from the [Magellan](https://sites.google.com/site/anhaidgroup/useful-stuff/data#TOC-The-784-Data-Sets-for-EM) project.

* Semi-Text: This task is for matching between semi-structued and unstructured tables. The source is from the Watch (-w) and Computer (-c) tasks from the [WDC Product Data](http://webdatacommons.org/largescaleproductcorpus/v2/index.html).

* Rel-Text: This task is for matching between structued and unstructured tables. The source is from the DBLP-ACM task from the [Deep Matcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) datasets. The table ACM is replaced with the abstract of the publication.

For more details about the pre- and post-processing of the datasets, please refer to Section 3 and Appendix of [our technique report](https://arxiv.org/abs/2106.08455).

## Data Format
There are 7 benchmarking tasks in total. The name of the folder is corresponding to the task described in the paper. There are 5 files in each dataset folder:
- left: The left table of entities.
- right: The right table of entities.
- train.csv: The training set of pairs
- valid.csv: The validation set of pairs
- test.csv: The test set of pairs

Each table contains an array of records. Based on the format of entities in the left/right tables, the suffix of them can be `.csv` (structured), `.json` (semi-structured) or `.txt` (unstructured).

The instances in train/valid/test files are triplets separated by `\t`: The first item is the id of an entity in the left table, where id means the subscription in the array of records; The second item is the id of an entity in the right table; The third item is the ground truth label (0/1).

## Statistics of Tasks

| Name | Left #Row | Left #Attr | Right #Row | Right #Attr | Train | Valid | Test | % Positive |
|------|-----------|------------|------------|-------------|-------|------|-------|--------------|
| Rel-HETER | 534 | 6.00 | 332 | 7.00 | 567 | 190 | 189 | 11.63 |
| Semi-HOMO | 2616 | 8.65 | 64263 | 7.34 | 17223 | 5742 | 5742 | 18.63 |
| Semi-HETER | 22133 | 12.28 | 23264 | 12.03 | 1240 | 414 | 414 | 38.2 |
| Semi-Rel | 29180 | 8.00 | 32823 | 13.81 | 1309 | 437 | 437 | 41.64 |
| Semi-Text-c | 9234 | 10.00 | 9234 | 1.00 | 5540 | 1846 | 1846 | 11.8 |
| Semi-Text-w | 20897 | 10.00 | 20897 | 1.00 | 12538 | 4180 | 4179 | 14.07 |
| Rel-Text | 2616 | 1.00 | 2295 | 6.00 | 7417 | 2473 | 2473 | 17.96 |

Left/Right #Row denotes the number of rows in the left/right table. Left/Right #Attr means the average number of attributes in the left/right table. Note that the number of attributes in an unstructured table is 1; the number of attributes for different records in a semi-structured table might be different from each other.
## Benchmarking Results

We evaluated 2 traditional machine learning based approaches: SVM and Random Forest; and 5 deep learning approaches: [DeepER](http://www.vldb.org/pvldb/vol11/p1454-ebraheem.pdf), [DeepMatcher](https://dl.acm.org/doi/10.1145/3183713.3196926), [Transformer](https://openproceedings.org/2020/conf/edbt/paper_205.pdf), [SentenceBert](https://aclanthology.org/D19-1410.pdf) and [Ditto](http://www.vldb.org/pvldb/vol14/p50-li.pdf) on the proposed tasks. The results of F1 score are as following.

| | SVM | Random Forest | DeepER | DeepMatcher | Transformer | SentenceBert | Ditto |
|-|-----|---------------|--------|-------------|-------------|--------------|-------|
| Rel-HETER | 0.821 | 0.706 | 0.872 | 0.936 | 0.955 | 0.696 | 1.00 | 
| Semi-HOMO | 0.53 | 0.747 | 0.875 | 0.861 | 0.938 | 0.874 | 0.931 | 
| Semi-HETER | 0.274 | 0.262 | 0.282 | 0.291 | 0.46 | 0.697 | 0.616 | 
| Semi-Rel | 0.709 | 0.733 | 0.436 | 0.567 | 0.905 | 0.59 | 0.911 | 
| Semi-Text-c | 0.557 | 0.65 | 0.418 | 0.442 | 0.886 | 0.798 | 0.818 | 
| Semi-Text-w | 0.556 | 0.505 | 0.388 | 0.427 | 0.665 | 0.502 | 0.649 | 
| Rel-Text | 0.436 | 0.363 | 0.529 | 0.534 | 0.631 | 0.329 | 0.627 | 


## Citation
If you are using the dataset, please cite the following in your work:
```
@inproceedings{cikm21machamp,
  author    = {Jin Wang and
               Yuliang Li and
               Wataru Hirota},
  title     = {Machamp: {A} Generalized Entity Matching Benchmark},
  booktitle = {CIKM},
  year      = {2021}
}
```

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses. In the event of conflicts between Megagon Labs, Inc. Recruit Co., Ltd., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. 
You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. 
All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein.
While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.

All datasets used within the product are listed below (including their copyright holders and the license conditions).
For Datasets having different portions released under different licenses, please refer to the included source link specified for each of the respective datasets for identifications of dataset files released under the identified licenses.

| ID | Dataset | Modified | Copyright Holder | Source Link | License | 
|------|-----------|------------|------------|-------------|-------|
| 1 | Fodor's and Zagat's restaurant | Yes | University of Texas | [source](https://www.cs.utexas.edu/users/ml/riddle/data/restaurant.tar.gz) | BSD 3-Clause |
| 2 | Citations | Yes |  University of Leipzig | [source](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution) | BSD 3-Clause |
| 3 | Magellan Book and Movie | Yes | University of Wiscosin Madison | [source](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository) | N/A |
| 4 | WDC Product Data | Yes | Web Data Commons | [source](http://webdatacommons.org/largescaleproductcorpus/v2/index.html) | Common Crawl Terms of Use |

