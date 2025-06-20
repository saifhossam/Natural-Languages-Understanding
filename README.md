# Aspect-Based Sentiment Analysis for Customer Reviews

This project focuses on extracting product aspects (e.g., "battery", "camera") from customer reviews and classifying their sentiment polarity (positive, negative, or neutral). It is particularly useful for analyzing e-commerce feedback to highlight product strengths and weaknesses.

---

**Dataset**

* **Source**: SemEval-2014 Task 4 (Laptops & Restaurants)
* **Format**: PyABSA-style XML files
* **Example**:

```xml
<sentence id="2339">
  <text>I charge it at night and skip taking the cord with me because of the good battery life.</text>
  <aspectTerms>
    <aspectTerm term="cord" polarity="neutral" from="41" to="45"/>
  </aspectTerms>
</sentence>
```

* **Processed Input Format**:

  * Tokenized format with IOB labels for aspect terms
  * Sentiment classes: {positive: 5330, negative: 2510, neutral: 1593}
  * ATE: 2122 train / 236 eval samples
  * ASC: 6665 train / 741 eval samples

---

**Models & Fine-Tuning: LoRA**

* **Technique**: LoRA (Low-Rank Adaptation)

  * Freezes base model
  * Trains only small injected layers → faster + memory-efficient
* **ATE Model**: `BertForTokenClassification`
* **ASC Model**: `AutoModelForSequenceClassification`
* **LoRA Config**: `r=8, lora_alpha=32, lora_dropout=0.1`

**Advantages of LoRA**

* Preserves semantic knowledge
* Trains 2.1× faster
* <1% accuracy drop compared to full fine-tuning (F1 = 0.82 vs 0.83)

---

**Evaluation Metrics**

* **ATE**:

  * Precision, Recall, F1 (binary averaging)
  * Token-level accuracy
* **ASC**:

  * Precision, Recall, F1 (weighted averaging)
  * Confusion matrix for polarity classification

---

**Results**

* Normal vs. LoRA fine-tuning performance compared
* Training/evaluation plots provided
* Confusion matrix visualized for ASC performance

---

**Structured Pipeline Output**

* **Aspect Term Extraction**:

```python
Sentence: "The pizza was delicious but the service was bad"
Tokens: ["the", "pizza", ..., "bad"]
Labels: ["O", "B-ASPECT", ..., "O"]
Output: ['pizza', 'service']
```

* **Aspect Sentiment Classification**:

```python
Output: {'pizza': 'positive', 'service': 'negative'}
```

---

**Team Members**

1. Aly Mohammed Aly (20210581)
2. Seif Hossam Aldin (20210441)
3. Sherif Ashraf Awad (20210453)
4. Abdelrahman Khaled (20210503)
5. Shrouk Mohammed (20210449)

---

**Key Takeaways**

* Efficient aspect-based sentiment classification using LoRA fine-tuning
* Real-world applicability in product review mining
* Strong results with reduced resource consumption
