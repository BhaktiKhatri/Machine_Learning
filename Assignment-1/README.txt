COMPARISON OF ID3 AND RANDOM VARIABLE SELECTION

AUTHORED BY :
Shalini Hemachandran (sxh163230)
Bhakti Khatri (brk160030)

Language Used : JAVA

Assuming jdk is installed, the following is the procedure to compile and execute the code.

Type following instructions in Command Prompt:

1) Change current directory in the command prompt to the directory where DecisionTreeCreatorBonus.java is placed
2) Type the following commands
		
		javac -d . DecisionTreeCreatorBonus.java
		java ml.decisiontree.id3.impl.DecisionTreeCreatorBonus <ABSOLUTE_PATH_OF_TRAINING_DATA_LOCATION> <ABSOLUTE_PATH_OF_TEST_DATA_LOCATION>

		Example :     1.
 				- javac -d . DecisionTreeCreatorBonus.java
				- java ml.decisiontree.id3.impl.DecisionTreeCreatorBonus C:\data\train-win.dat C:\data\test-win.dat
	      		      2.
				- javac -d . DecisionTreeCreatorBonus.java
				- java ml.decisiontree.id3.impl.DecisionTreeCreatorBonus C:\data\train2-win.dat C:\data\test2-win.dat
		
	       

The code has been tested with train2/test2 datasets