# text-Comparison

TextCompare is a Python library for comparing and analyzing text similarity. It provides various metrics and pipelines for text comparison tasks.

### Detect Temporal shift
We captures the specific association between a time and an action.This captures the specific association between a time and an action.
We use sets to compare these temporal-action pairs between the two texts. This allows us to identify exact matches in time-action associations.
We calculate the Jaccard similarity between these sets, which gives us a measure of how similar the temporal-action structures are between the two texts.

```python
text1 = "Yesterday, I stayed at the store. Today, I'm staying home."
text2 = "Today, I stayed at the store. Yesterday I stayed home."
```

```python
Temporal actions in text 1: [('yesterday', 'stay'), ('today', 'stay')]
Temporal actions in text 2: [('today', 'stay'), ('yesterday', 'stay')]
Matching temporal-action pairs: set()
Total unique temporal-action pairs: {('yesterday', 'stay'), ('today', 'stay')}
Temporal shift similarity: 0.0
Final similarity score: 0.0

```

```python
text3 = "Yesterday, I went to the store. Today, I'm staying home."
text4 = "Today, I went to the store. Yesterday I stayed home."
```
```python
Temporal actions in text 3: [('yesterday', 'go'), ('today', 'stay')]
Temporal actions in text 4: [('today', 'go'), ('yesterday', 'stay')]
Matching temporal-action pairs: set()
Total unique temporal-action pairs: {('yesterday', 'go'), ('today', 'stay'), ('today', 'go'), ('yesterday', 'stay')}
Temporal shift similarity: 0.0
Final similarity score: 0.0
```





