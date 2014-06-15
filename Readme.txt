The files under this directory are for the Final Project of the course EMNLP.

This directory contains
    Executable jar files:   whatswrong-0.2.3-standalone.jar
                            conll08-eval.jar
    Plain text files:       trn.conll08
                            dev.conll08

Final Project: Transition-Based Dependency Parsing

Schedule
    Train and devlopment data availible:    May 22, 2014
    Test data availible:                    June 22, 2014
    System submission due:                  June 29, 2014
    Report submission due:                  June 29, 2014
    
Tasks Description
    Main task:  
        Build a transition-based dependency tree parser.
        You are allowed to use the Liblinear library
        (http://www.csie.ntu.edu.tw/~cjlin/liblinear/).
    Optional task: 
        1. Instead of using the Liblinear library, build a classifier.
        2. Build a transition-based dependency graph parser.
        
Train and Development Data
    The train and development data are in .conll08 format.
    Sentences are seperated by empty lines.
    Each sentence contains several columns:
        column 1:   index;
        column 2-3: token;
        column 4-5: pos;
        column 6-8: leave out;
        column 9:   tree head index;
        column 10:  tree arc label (all are 'X');
        column 11:  graph heads;
        column 12+: graph dependents, each column for one head in column 11.
        
Example for .conll08 format:
1	Ms.	Ms.	NNP	NNP	_	_	_	2	X	Ms.	_	_
2	Haag	Haag	NNP	NNP	_	_	_	3	X	_	X	X
3	plays	plays	VBZ	VBZ	_	_	_	0	X	plays	_	_
4	Elianti	Elianti	NNP	NNP	_	_	_	3	X	_	_	X
5	.	.	.	.	_	_	_	3	X	_	_	_
    In the above sentence, there are 5 tokens.
    The tree on the sentence is given by column 9 and column 10.
    The column 9 and column 10 of line 1 indicates that the head of 'Ms.' is
    'Haag'; the two columns of line 3 indicates that the head of 'plays' is the
    root.
    The graph on the sentence is given by column 11-13.
    The column 11 of the sentence indicates that there are two heads in the
    graph, say, 'Ms.' and 'plays'; the column 12 indicates that the only
    dependent of 'Ms.' is 'Haag'; the column 13 indicates that the dependents
    of 'plays' are 'Haag' and 'Elianti'.
    You can run whatswrong-0.2.3-standalone.jar to visualize the trees and
    graphs.

Test Data
    Only first 5 columns of the .conll08 file are provided.

Evaluation
    For dependency tree parsing, unlabeled accuracy and complete match are
    considered;
    for dependency graph parsing, unlabeled precision, recall, F-messure and 
    complete match are considered.
    You can run conll08-eval.jar to evaluate your result on the development
    data.

Requrements:
    1.  Build your own system. (Anyone caught cheating will not receive credit)
    2.  Submit your system and report in time. (Your credits will be penalized
        by 30% if your submission is late)
    3.  Your submission of system is supposed to include source codes,
        excecutables, results in the format descripted above, and a simple
        readme file.
    4.  Your report is supposed to be a .pdf file, in formal paper format, 
        demonstrating your idea, algorithm, implementation, experiments,
        and other ralated information. Your report is supposed to be at most 4 
        pages.
