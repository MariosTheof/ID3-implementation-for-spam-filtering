# ID3 implementation for spam filtering
An ID3-tree algorithm implementation which classifies emails as ham or spam, using the enron database.
It was developed using spyder-IDE.

You can find the enron database [here](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html).

For the classification we use binary properties. eg. Whether a common spam word is contained in the email.

You can use pprint to view your tree structure.
This:
```python
    import pprint
    pprint.pprint(tree)
```
Should give you something like :
