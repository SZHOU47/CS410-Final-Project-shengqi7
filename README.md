# CS410-Final-Project-shengqi7

# Emerging Financial Complaints Radar – Windows User Setup & Usage

## This tool is a command-line Python app for exploring CFPB consumer complaint narratives.

The script has two interactive parts:


*	Query-based word association discovery (Part 1)
*	Complaint topic trends over time (Part 2)


Everything is handled by a single script:
```
project.py
```
For windows user:
*	You will need to have Python, Pyserini, Ubuntu, and WSL installed on your laptop to be able to run the program.
*	Data/Folder Setup - Put the CFPB complaints csv data in the home/[your username] folder under a data folder, and leave the project.py and requirements.txt in the home/[your username] folder
*	Ensure the following packages are installed by running requirements.txt:

  ```
  pyserini==1.2.0
  python-dotenv==1.1.1
  pandas==2.3.2
  numpy==1.26.4
  matplotlib==3.10.6
  tqdm
  ```

On the first run, the script will:

* Create a search corpus : .\data\search_corpus.tsv
* Preprocess for Pyserini : .\data\preprocessed_corpus\
* Build a Lucene index : .\data\complaints_index\
  
On later runs, if preprocessed_corpus and complaints_index already exist, the script will reuse them and skip rebuilding.

Running the Tool on Windows:

From the home/[your username] folder (where your script lives):

Ensure your dataset is placed under the data folder, then run:

Run: 
```
python project.py
```
