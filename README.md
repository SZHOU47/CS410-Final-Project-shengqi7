# CS410-Final-Project-shengqi7

# Emerging Financial Complaints Radar – Windows User Setup & Usage

## This tool is a command-line Python app for exploring CFPB consumer complaint narratives.

The script has two interactive parts:


1. **Query-based word association (Part 1)**  
   - You type a search query (e.g., `identity theft`).  
   - The tool retrieves the most relevant complaints using **BM25 + RM3** (via Pyserini).  
   - It then mines **key phrases** that frequently co-occur with your query in those complaints.  
   - Finally, it prints the **top retrieved complaints** with metadata and nicely formatted narratives.

2. **Complaint topics trends over time (Part 2)**  
   - You choose a date range and optional company filter.  
   - The tool discovers **topic-like phrases** across all complaints in that slice.  
   - Then it builds a **monthly trend table** and **line chart** showing how often each phrase appears over time.


Everything is handled by a single script:
```
project.py
```
For windows user:
*	You will need to have Python, Pyserini, Ubuntu, and WSL installed on your windows laptop to be able to run the program.

Follow this link to install WSL: 

https://learn.microsoft.com/en-us/windows/wsl/install

Follow this link to install pyserini:

https://github.com/castorini/pyserini/blob/master/docs/installation.md#pypi-installation-walkthrough

Follow this link to install Anaconda:

https://docs.anaconda.com/anaconda/install/linux/
  
*	Data/Folder Setup - Put the CFPB complaints csv data in the home/[your username]/data folder, and leave the project.py and requirements.txt in the home/[your username] folder
*	Ensure the following packages are installed by running requirements.txt:

  ```
  pyserini==1.2.0
  python-dotenv==1.1.1
  pandas==2.3.2
  numpy==1.26.4
  matplotlib==3.10.6
  tqdm
  ```

Run:

```
python -r requirements.txt
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












