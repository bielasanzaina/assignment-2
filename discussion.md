
# Assignment 2 Discussion

## Problem 1(B)

Based only on the prior probabilities, I would be more likely to predict Johnson as the author of an unlabeled speech. Johnson has 66 of the 102 training documents, so P(Johnson) ≈ 0.647, while Kennedy has 36 of 102, so P(Kennedy) ≈ 0.353.

## Problem 1(C)

Johnson's documents tend to be longer than Kennedy's in this dataset. Kennedy's speeches contain about 2,879 words on average, while Johnson's contain about 3,620 words on average.

The most frequent words for both authors are mostly common function words such as "the," "of," "and," "to," and "in." Kennedy's top ten also includes "our," while Johnson's includes "I." Because Johnson has more documents and longer speeches overall, the raw word counts should not be directly compared as usage rates.

Both authors often begin their speeches with formal greetings or direct addresses to the audience. Kennedy frequently starts by addressing officials or institutions, while Johnson also commonly begins with greetings such as “Good evening, my fellow Americans” or formal congressional addresses.

## Problem 1(E)

The estimated prior probabilities are approximately 0.353 for Kennedy and 0.647 for Johnson.

The likelihood matrix has shape `(2, 24390)`, with one row for each author class and one column for each vocabulary item.

As alpha increases, Lidstone smoothing becomes stronger, so the likelihood estimates are influenced less by the raw observed counts and are smoothed more heavily. For example, the likelihoods for the word "the" decreased for both authors as alpha increased from 0.01 to 1.0.

## Problem 1(F)

The custom Naive Bayes classifier predicted the ten unlabeled speeches as:

1. Kennedy
2. Kennedy
3. Kennedy
4. Johnson
5. Kennedy
6. Kennedy
7. Kennedy
8. Johnson
9. Johnson
10. Kennedy

## Problem 2(A)

The scikit-learn Naive Bayes classifier produced almost the same predictions as the custom implementation. The two models differed on only one of the ten test documents. The custom model predicted the seventh document as Kennedy, while scikit-learn predicted it as Johnson.

## Problem 3(A)

The custom Naive Bayes classifier achieved an accuracy of 0.80 and an F1-score of 0.75.

The scikit-learn Multinomial Naive Bayes classifier achieved an accuracy of 0.90 and an F1-score of approximately 0.889.

## Problem 3(B)

For the custom classifier, the confusion matrix was:

[[5, 0],
 [2, 3]]

The model correctly classified all five Kennedy speeches, but it misclassified two Johnson speeches as Kennedy.

For the scikit-learn classifier, the confusion matrix was:

[[5, 0],
 [1, 4]]

The scikit-learn model also classified all five Kennedy speeches correctly, but it misclassified only one Johnson speech as Kennedy. This is consistent with its higher accuracy and F1-score. These results are visualized in `conf.jpg`, which shows the confusion matrices for both classifiers side by side.