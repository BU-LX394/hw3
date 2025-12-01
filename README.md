# HW 3: Spam vs. Ham

**Due: December 11, 9:00 AM**

For this assignment, please complete the _problem set_ found in `hw3-pset.ipynb`. The problem set includes coding
problems as well as written problems.

For the coding problems, you will implement functions defined in the Python file `hw3.py`, replacing the existing code (
which raises a `NotImplementedError`) with your own code. **Please write all code within the relevant function
definitions**; failure to do this may break the rest of your code.

For the written problems, please submit your answers in PDF format, using the filename `hw3-written.pdf`. Make sure to
clearly mark each problem in order to minimize the chances of grading errors.

You do not need to submit anything for Problems 0, 1c, 1f, or 3a.

You are free to use any resources to help you with this assignment, including books, websites, or AI assistants such as
ChatGPT (as long as such assistants are fully text-based). You are also free to collaborate with any other student or
students in this class. However, you must write and submit your own answers to the problems, and merely copying another
student's answer will be considered cheating on the part of both students. If you choose to collaborate, please list
your collaborators at the top of your `hw3-written.pdf` file. If you choose to use ChatGPT or a similar AI assistant to
help you with this assignment in any way, please include a file called `hw3-chatgpt-logs.txt` with a full transcript of
all prompts and responses used for this assignment. Your use of AI assistants cannot involve generating images or any
other content that cannot be included in a `.txt` file.

## Setup

You will need to complete your code problems in Python 3, preferably Python 3.8 or later. The problem set itself is a
Jupyter notebook; it is highly recommended that you are able to run notebooks on your own computer in order to complete
this assignment.

We will not be using any Python libraries apart from the standard libraries and [NLTK](https://www.nltk.org/).

Before starting the assignment, you will need to download
the [Enron Spam Dataset](https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data/data)
[(Metsis et al., 2006)](https://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf) from the Assignments folder of
Blackboard and put the `enron_spam_data.csv` file in the `data` folder. Further setup instructions are given in Problem 0.

## Submission

For your submission, please upload the following files to [Gradescope](https://www.gradescope.com):

* `hw3.py`
* `hw3-written.pdf`
* `hw3-chatgpt-logs.txt` (if you used ChatGPT or a similar text-based AI assistant)

Do not change the names of these files, and do not upload any other files to Gradescope. Failure to follow the
submission instructions may result in a penalty of up to 1 point.

## Grading

The point values for each problem are given below. 1 point is given for free, for Problem 3a.

| Problem                           | Problem Type  | Points |
|-----------------------------------|---------------|--------|
| 1a: Understand Dict Disjunction   | Written       | 1      |
| 1b: Understand CSV Files          | Written       | 1      |
| 1d: Skip Header Row               | Written       | 1      |
| 1e: Load Data from CSV File       | Code          | 1      |
| 1g: Understand Variable Unpacking | Written       | 1      |
| 2a: Deserialize Spam Dataset      | Code          | 2      |
| 2b: Extract Common Token Types    | Code          | 2      |
| 2c: Extract Features from Emails  | Code          | 2      |
| 2d: Preprocess Datasets           | Code          | 2      |
| 2e: Evaluate Spam Classifier      | Code          | 4      |
| 3a: Run Experiment                | No Submission | 1      |
| 3b: Report Results                | Written       | 1      |
| 3c: Analyze Results               | Written       | 1      |
| **Total**                         |               | **20** |

### Rubric for Code Problems

Code questions will be graded using a series of [Python unit tests](https://realpython.com/python-testing/). Each
function you implement will be tested on a number of randomly generated inputs, which will match the specifications
described in the function docstrings.

For code questions, you will receive:

* full points if your code runs and passes all test cases
* at least .25 points if your code runs but fails at least one test case
* 0 points if your code does not run.

Partial credit may be awarded at the Yulu's discretion, depending on the correctness of your logic and the severity of
bugs or other mistakes in your code. All code problems will be graded **as if all other code problems had been answered
correctly**. Therefore, an incorrect implementation of one function should (in theory) not affect your grade on other
problems that depend on that function.

### Rubric for Written Problems

For written problems, you will receive:

* full points if your answer is completely correct
* at least .25 points if a good-faith effort (according to Yulu's judgment) has been made to answer the question
* 0 points if your answer is blank.

Partial credit may be awarded at Yulu's discretion.

## Late Submissions and Resubmissions

Grading will commence on December 18, and solutions will be released on that day. Therefore, no late submissions will be
accepted after 9:00 AM on December 18. You may resubmit your solutions as many times as you like; only the final
submission will be graded. If the final submission occurs after the deadline on December 11, then your submission will
be considered late even if you have previously submitted your solution before the deadline.