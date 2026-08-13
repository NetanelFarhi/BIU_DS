# REPORT — Module 3 · Assignment 2 · Unsupervised Learning

**Name:** ___ 
**Chosen option:** B · Credit Card behavioral segmentation

---

## 1. Framing
I looked for behavioral customer segments in credit card usage data. The business goal is to support different marketing, risk, and activation actions for different customer types.

Distance measure: Euclidean distance after preprocessing. I used it because KMeans and Ward clustering are distance-based, and the features are numeric. Many variables were skewed and correlated, so I compared raw scaling with log-based preprocessing and used a reduced log-based feature set.

---

## 2. Method & validation

| Item | Value |
|---|---|
| Approaches tried | KMeans + Ward hierarchical clustering |
| Final preprocessing | reduced_log_scaled |
| Chosen k | 4 |
| Silhouette score | 0.2212 |
| Ward vs KMeans ARI | 0.5683 |
| Mean subsample ARI | 0.9856 |
| Largest holdout share difference | 0.0158 |
| Weakest cluster | Cluster 2 |

---

## 3. Guiding questions

1. **No ground truth.** I judged the clustering using internal metrics, cluster sizes, stability checks, PCA visualization, and whether the profiles made business sense. This evidence is limited because there are no labels proving that these are the true customer groups.

2. **Choosing k.** In the final setup, Elbow and Silhouette both supported k=4. I also preferred it because it avoided one broad “everyone else” cluster and gave more usable business segments.

3. **Scaling.** Scaling changed the distance space and the resulting clusters. The unscaled Silhouette is not directly comparable because it is measured in a different space and is dominated by large monetary columns.

4. **Stability.** Seed stability was high, but that mainly shows the optimizer is stable. The better evidence is the subsample and holdout checks, which test whether similar structure appears when the data changes.

5. **What defines each cluster.** The main separating features are shown in the segment card. Overall, the clusters are separated by purchase behavior, cash advances, balance/payment behavior, and activity level.

6. **Real or artifact.** Ward and KMeans had moderate agreement, so the result is not completely algorithm-independent. Cluster 2 is the weakest segment by Silhouette and should be treated as more tentative than the others.

7. **Action.** Each segment has one suggested business action below. If a segment is too broad or unstable, it should be used carefully or split further for a specific campaign.

8. **Cost of a false alarm.** For clustering, the risk is targeting customers with the wrong action. That can waste campaign budget or create a bad customer experience, so the segments should support decisions rather than automate them blindly.

---

## 4. Structure Card

```text
STRUCTURE CARD — Credit Card Behavioral Segmentation
Final preprocessing: reduced_log_scaled
Final k: 4
Silhouette: 0.2212
Ward vs KMeans ARI: 0.5683

Cluster 0 | 36.6% | High-balance non-purchasers with cash-advance use | Defined by: CASH_ADVANCE (above average), CASH_ADVANCE_FREQUENCY (above average), PURCHASES_FREQUENCY (below average), INSTALLMENTS_PURCHASES (below average) | Action: Monitor risk and offer cheaper repayment or credit-planning options.
Cluster 1 | 25.7% | Large one-off purchase customers | Defined by: ONEOFF_PURCHASES_FREQUENCY (above average), ONEOFF_PURCHASES (above average) | Action: Offer targeted rewards around large purchases and premium card usage.
Cluster 2 | 14.9% | Low activity customers | Defined by: BALANCE_FREQUENCY (below average), BALANCE (below average) | Action: Use low-cost digital activation offers before spending retention budget.
Cluster 3 | 22.8% | Installment-oriented purchasers | Defined by: PURCHASES_INSTALLMENTS_FREQUENCY (above average), INSTALLMENTS_PURCHASES (above average), CASH_ADVANCE (below average), ONEOFF_PURCHASES (below average) | Action: Promote structured installment offers and merchant campaigns.
```

---

## 5. Reflection
The main surprise was how much preprocessing mattered. Standard scaling alone was not enough because the data had strong skewness and redundant behavior variables. I would trust the segments as a useful first business view, but I would validate them on newer data before using them for a real campaign.
