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

   1. Follow this link to install WSL: 

      https://learn.microsoft.com/en-us/windows/wsl/install

   2. Follow this link to install pyserini:

      https://github.com/castorini/pyserini/blob/master/docs/installation.md#pypi-installation-walkthrough

   3. Follow this link to install Anaconda:

      https://docs.anaconda.com/anaconda/install/linux/
  
*	Data/Folder Setup - In the Linux/Ubuntu environment, put the complaints_data.csv data in the home/[your username]/data folder, and leave the project.py and requirements.txt in the home/[your username] folder
  
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

## Running the software on Windows:

Ensure your dataset is placed under the data folder and the data folder is in the same place as the python file:

1. First run:
```
conda activate pyserini
```

2. Then run to start the software: 
```
python project.py
```

The program takes few seconds to load and on startup you’ll see:

A banner: FINANCIAL COMPLAINTS EXPLORER

A message saying you’ll run Part 1 (query-based word association), then Part 2 (topic trends).

<img width="406" height="143" alt="image" src="https://github.com/user-attachments/assets/51b37ffa-86f7-4cf2-8c5a-6574a53cc4c5" />

* In Part 1:

You will be asked to enter the words you want to search and an optional year filter:

For example:

<img width="410" height="473" alt="image" src="https://github.com/user-attachments/assets/02bcb8bf-e479-496e-a115-989bc1cdc682" />

* In Part 2:

You will be asked to enter the start date and end date for the desired date range, and you will be asked to enter an optional company/bank filter, and at last, a number of topics to plot filter:

For example:

<img width="565" height="507" alt="image" src="https://github.com/user-attachments/assets/7f1ef390-0e8a-4d0b-9d3d-d8daf9e73cdd" />

At the end, a windows with chart of showing monthly trends would pop up:

For example:

<img width="493" height="234" alt="image" src="https://github.com/user-attachments/assets/1b29524d-72e4-490f-b6e4-9753567b727e" />










