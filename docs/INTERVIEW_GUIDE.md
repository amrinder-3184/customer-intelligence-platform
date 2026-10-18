# Interview guide: Customer Intelligence Platform

A study guide for explaining this project in Data Science and Software Engineering interviews. Each section covers what was built, *why*, and questions you should be able to answer. Numbers come from the saved reports (see the README's results table, which a test keeps in sync).

**The 30-second pitch:** "I built an end-to-end churn platform on the IBM Telco dataset. I validated the data, analysed churn with pandas and SQL (cross-checked against each other), and built a leakage-safe scikit-learn pipeline. Logistic regression was kept after a fair comparison with tree models, and I chose a cost-based threshold from training data only. The model is served through a validated FastAPI service. The held-out test set was protected by code, not just discipline, and one command verifies the whole project's integrity."

---

## Phase 1: data ingestion and validation

**What:** download the CSV reproducibly (with a SHA-256), read every column as text, and check the schema, categories, ranges, duplicates and cross-column rules.

**Key ideas**
- **Read as text first** (`dtype=str, keep_default_na=False`). Pandas otherwise guesses types and silently turns `" "` into NaN, hiding the 11 blank `TotalCharges` values.
- **Errors vs warnings.** An error means the data can't be trusted (unknown category, duplicate ID). A warning is a known quirk to handle deliberately (blank `TotalCharges`, all at tenure 0).
- **Checksums** prove that you and a reviewer are using byte-identical data.
- **Class imbalance:** 26.54% churn, so "always predict no churn" is 73.5% accurate and useless.

**Questions**
1. *Why did `TotalCharges` load as text?* Eleven values are a single space. Filling them with the column mean would invent bills for customers who haven't been billed yet (all 11 have tenure 0).
2. *Why is a duplicate customerID an error, but a duplicate feature profile only a warning?* Two different customers can have the same plan, but one customer appearing twice could leak between train and test.
3. *Why test a validator with deliberately broken data?* A validator that always passes is useless. Each test breaks one rule and expects that error.

## Phase 2: EDA and SQL

**What:** churn rates by segment with 95% Wilson intervals; a SQLite view that types the data *independently* of pandas; 7 SQL queries cross-checked against pandas.

**Key ideas**
- **Independent cross-check:** SQL and pandas compute the same numbers separately. Deliberately tampering with one result proves the comparison can fail.
- **Association, not causation:** month-to-month contracts churn at 42.7% vs 2.8% for two-year contracts, but customers *choose* contracts.
- **Confounding:** fiber churns more than DSL *within every contract type*. That reduces, but doesn't remove, the chance that one factor is just a proxy for the other.
- **Snapshot bias in tenure:** a churned customer's tenure stops at churn, while a retained customer's keeps growing.
- **Safe SQL:** fixed, read-only queries; values go in through `?` placeholders, never string formatting.

**Questions**
1. *Would moving everyone to two-year contracts cut churn 15×?* No. That's a causal claim from observational data. Customers who commit for two years probably differ in ways the data doesn't record.
2. *Why build the typed view in SQL instead of loading pandas' cleaned data?* Then the two calculations are independent, so agreement means something.
3. *Why Wilson intervals?* They stay within [0, 1] and behave better than the simple ±1.96·SE formula for small groups.

## Phase 3: leakage-safe baseline

**What:** a stratified 80/20 split (seed 42) before any fitting; all learned preprocessing inside one `Pipeline`; a Dummy baseline vs logistic regression; 5-fold CV on training data; the test set scored once at 0.5.

**Key ideas**
- **Stateless vs learned steps.** Converting `" "` to NaN learns nothing, so it's safe before the split. A median or a scaler mean *is learned*, so it must come from training rows only.
- **Pipeline + CV:** the whole pipeline is re-fitted inside each fold, so validation folds never influence preprocessing.
- **Accuracy paradox:** the Dummy gets 0.735 accuracy and catches no churners. LogReg's 0.806 is only 7 points better, but its ROC-AUC is 0.842.
- **ROC-AUC vs PR-AUC:** ROC-AUC measures ranking, with a no-skill level of 0.5. PR-AUC focuses on the positive class, and its no-skill level is the churn rate, 0.265.
- **`class_weight="balanced"`** barely changes ranking (same ROC-AUC). It mostly shifts where 0.5 falls: higher recall, lower precision, and worse probabilities.
- **Retrospective leakage:** `tenure` and `TotalCharges` are legal columns, but for churners they're measured *at* churn. Without them, CV ROC-AUC falls from 0.846 to 0.832.

**Questions**
1. *Why is computing the `TotalCharges` median before the split leakage, while converting to numeric isn't?*
2. *Why use the test set only once?* Each time you look at it and change something, it becomes part of training, and the estimate becomes optimistic.
3. *13 test customers share a profile with a training customer. Is that leakage?* No. They're different customers, and excluding them barely changes ROC-AUC (0.8423 → 0.8431).

## Phase 4: model comparison, threshold, calibration, interpretation

**What:** out-of-fold (OOF) predictions on training data for every decision, with pre-declared decision rules, then one locked test evaluation.

**Key ideas**
- **Paired fold comparison:** the random forest beat LogReg by +0.0017 ROC-AUC with a standard error of 0.0033 (3/5 folds), which is noise. Folds share training rows, so this SE is optimistic. Use it as a sanity check, not a significance test.
- **Calibration vs ranking:** a model can rank perfectly yet output poor probabilities. Brier and log loss are *proper scoring rules*. ECE depends on how the bins are chosen. LogReg's ECE was 0.010, and calibrators didn't improve Brier or log loss.
- **Nested calibration:** `CalibratedClassifierCV` uses its own inner CV inside each outer fold, so calibrators never see the labels they're evaluated on (tested with a recording estimator).
- **Cost-optimal threshold:** flag a customer when p > FP/(FP+FN). With FN = 5×FP, that's 1/6 ≈ 0.167; the data chose 0.16. It works only if the probabilities are calibrated, and the theoretical value fell inside the near-optimal band for every cost ratio tested.
- **The threshold is a business decision:** the model supplies probabilities, and costs decide the action.
- **Coefficients with correlated features:** `TotalCharges` has a *positive* coefficient although churners have *lower* totals. Each coefficient holds the other features fixed, and tenure and TotalCharges are correlated at 0.83.
- **Permutation-importance trap:** shuffling `tenure` alone creates impossible customers (1 month of tenure with a $5,000 total), so it overstates importance. Shuffle correlated features *as a group*.
- **Test-set lock:** decisions are hashed and saved before the single test evaluation, and changed decisions are refused.

**Questions**
1. *The random forest scored 0.848 vs 0.846. Why not switch?*
2. *Isotonic calibration had the lowest ECE. Why wasn't it chosen?*
3. *If a missed churner costs 10× an offer, what threshold does theory suggest, and what does that assume?*
4. *Why can't you call `tenure` "the most important feature" from its single-feature permutation score?*

## Phase 5: model persistence and API

**What:** fit the selected pipeline on the verified training rows; save it with metadata (schema, threshold, provenance, versions); FastAPI with Pydantic validation; the model loaded once at startup.

**Key ideas**
- **Save the fitted pipeline, not just the model:** the imputer median, the scaler means and the category lists are learned values that inference must reuse.
- **Schema drift fails silently:** with `handle_unknown="ignore"`, the value `"0.0"` instead of `"0"` doesn't crash; it just becomes all zeros. A parity test (API path vs training path on 300 training rows) caught exactly this when injected deliberately.
- **Validation as a security boundary:** `extra="forbid"`, `Literal` categories, strict numbers, no NaN or Infinity, and cross-column rules → 422 before any model code runs.
- **Probability vs class:** the probability is the estimate; `predicted_churn` applies a threshold, which is policy stored in metadata.
- **Failing safely:** if the artifact is missing or corrupted, the service starts but `/health` returns 503, and no traceback or file path is ever returned.
- **Pickle safety:** a joblib file can execute code when loaded. Check its hash *before* loading, and never load an uploaded artifact.
- **A 200 response isn't model quality:** it shows the code path works. Usefulness needs monitoring and evaluation on future data.

**Questions**
1. *Why is pickling only the `LogisticRegression` step a bug?*
2. *Why return 503 instead of crashing when the model can't load?*
3. *Why not retrain on startup?* It's slow, non-reproducible, and risks training on the wrong data.
4. *How do you test model logic separately from HTTP behaviour?* The predictor class is tested directly with a fake pipeline (for example, the threshold edge at exactly 0.16); `TestClient` tests status codes and error bodies.

## Phase 6: reproducibility, integrity and documentation

**What:** `make_split` (a fresh clone can rebuild the split without scoring the test set), `verify` (one read-only command with 7 checks), committed reference hashes, exploratory output kept separate, and a README guarded by tests.

**Key ideas**
- **Fingerprints, not copies:** the split is described by hashes of the sorted train and test customerIDs, so a fresh clone can prove it rebuilt *the same* split without the file being committed.
- **Frozen results:** outputs of test-set evaluations are hashed once from trusted files. You never "fix" a failing hash by regenerating it; you investigate.
- **Line endings break byte hashes:** Git's CRLF↔LF conversion would change the hashed bytes on other machines, so `.gitattributes` marks `reports/**` as `-text`.
- **Read-only checks:** `verify` writes no files (tested by snapshotting every project file before and after) and makes exactly one prediction, the made-up API customer (tested by recording calls).
- **Docs that can't drift:** each number in the README results table is compared with its saved source file by a test.
- **Lock file vs pins:** `requirements.txt` pins the direct dependencies; `requirements-lock.txt` records every package version for exact reproduction.

**Questions**
1. *Why is `make_split` safe while `train_baseline` isn't, even though both produce the same split?*
2. *A `verify` hash fails after a teammate edits a report. What do you do?* Don't regenerate the hash. Find out what changed and why. If the change is legitimate, the frozen result has changed, and that needs review.
3. *How do you prove a check is read-only?* Snapshot the bytes and modification times of every file before and after, and compare.
4. *What does a passing `verify` prove, and what doesn't it prove?*

---

## Cross-cutting questions

1. **"What would you do with more time?"** Rebuild the dataset so features are frozen at a prediction date and churn is observed in a later window. Then get real retention costs, and monitor calibration and drift after deployment.
2. **"What was the hardest bug?"** A silent one: an encoder configured to ignore unknown categories turned a wrong input format into all-zero features without any error. Only a parity test comparing the API path with the training path caught it.
3. **"How did you prevent test-set leakage?"** Every learned step was inside a `Pipeline`, and every decision came from training-only out-of-fold predictions. The rules were pre-declared and the decisions hashed. A lock refuses re-scoring under changed decisions, and tests record that selection code never touches test rows.
4. **"Is your model production-ready?"** No. The software is tested and reproducible, but the model is retrospective, the costs are assumed, and there's no monitoring.
