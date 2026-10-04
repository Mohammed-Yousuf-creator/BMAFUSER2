## 1. README.md

I created a comprehensive developer README covering:

- Git/Conda setup
- official Conda download/install links
- `conda env create -f environment.yml`
- `conda activate climaFuse`
- repository/file structure
- AIFS downloader CLI
- GFS downloader CLI
- xESMF regridding CLI
- BMA training CLI
- BMA evaluation CLI
- direct `bma_fusion.py` usage
- `orchestrator.py` real-time usage
- `orchestrator.py --offline` usage
- `orchestrator_realtime.py`
- Python/backend integration
- troubleshooting
- temperature target semantics
- explanation of why BMA MAE can be worse than an individual model

:chatgpt-content-reference{index="4"}[Download the README.md](sandbox:/mnt/data/README.md)

### One important repository issue

I checked the current `main` branch of `BMAFUSER2`. It currently **does not contain `environment.yml`**, so this command:

```bash
conda env create -f environment.yml
```

cannot work for another developer until that file is committed to the repository.

The official Conda/Anaconda download is here: [Download Conda / Miniconda](https://www.anaconda.com/download?utm_source=chatgpt.com). Conda's documentation also recommends Miniconda as a lightweight installation option. [Conda Documentation](https://docs.conda.io/en/latest/miniconda.html?utm_source=chatgpt.com)

I also attempted to commit the README directly to GitHub, but the connected GitHub integration currently lacks repository write permission, so I could create the file but could not push it into `main`. You can add the downloaded file as `README.md` and commit it normally.

---

# 2. Why can BMA MAE be greater than both AIFS and GFS?

This is actually explainable from **your implementation**, and it does not automatically mean the BMA implementation is broken.

Your BMA fitting function minimizes the **negative log-likelihood of a Gaussian mixture**, not MAE:

\[
\theta^*=\arg\min_\theta -\sum_i \log
\left[
w_A f_A(y_i)+(1-w_A)f_G(y_i)
\right]
\]

So the optimization target is:

> **"Find the best probabilistic distribution."**

It is **not**:

> "Find the point forecast with minimum MAE."

Those are different optimization problems.

### The biggest reason in your code

For each location, BMA learns a global AIFS/GFS weight:

\[
w_A,\quad w_G=1-w_A
\]

and also learns calibration parameters:

\[
\mu_A=a_A+b_Ax_A
\]

\[
\mu_G=a_G+b_Gx_G
\]

Therefore your BMA expected temperature is approximately:

\[
E[BMA]
=
w_A(a_A+b_Ax_A)
+
w_G(a_G+b_Gx_G)
\]

That means the BMA forecast isn't necessarily equal to:

```text
weighted_average(AIFS, GFS)
```

It is a **calibrated weighted mixture**.

---

## Example from your actual results

New Delhi:

```text
AIFS MAE = 0.809 °C
GFS  MAE = 3.068 °C
BMA  MAE = 0.860 °C
```

Here AIFS is clearly the stronger deterministic forecast.

But BMA still assigns some probability/weight to the GFS component. If that component pulls the calibrated mixture mean away from AIFS, the resulting point forecast can have:

```text
0.860 > 0.809
```

while still being dramatically better than GFS.

That is completely possible.

---

## There is another important reason: training ≠ evaluation

Suppose during training you had:

```text
AIFS usually better
GFS sometimes better
```

BMA learns a compromise such as:

```text
AIFS weight = 0.75
GFS weight  = 0.25
```

But suppose during your held-out period AIFS happens to be overwhelmingly better.

The BMA model **doesn't know that yet**.

It is using the weights learned from the training period:

```text
Training data
     ↓
learn weights
     ↓
freeze model
     ↓
held-out evaluation
```

So on a particular test period, a fixed mixture can lose to whichever individual model happens to dominate that period.

---

# 3. Why BMA can still be better even when MAE is worse

This is the key distinction.

MAE evaluates a **point forecast**:

\[
MAE=\frac1N\sum |y_i-\hat y_i|
\]

BMA produces a **probability distribution**.

You therefore need to evaluate things such as:

- MAE
- RMSE
- bias
- interval coverage
- interval width
- likelihood / proper probabilistic score

Your evaluator already reports several of these.

For example, a BMA forecast might be:

```text
          probability
              ^
              |
        _____/ \_____
      _/           \_
----|-------------------|---- temperature
   28                  32
```

while AIFS might simply give:

```text
AIFS = 30.0 °C
```

A BMA model could have a slightly worse mean but a much more realistic uncertainty distribution.

So:

```text
Best MAE
```

and

```text
Best probabilistic forecast
```

are not necessarily the same model.

---

# 4. Your precipitation model has an additional issue

Your precipitation BMA uses:

```python
log1p(precipitation)
```

during fitting.

So the optimization occurs in transformed space:

\[
z=\log(1+y)
\]

and then you transform back:

\[
y=e^z-1
\]

The expectation on the original scale is therefore not simply:

\[
w_A x_A+w_Gx_G
\]

because of the nonlinear exponential transformation.

This can further separate **likelihood-optimal prediction** from **minimum raw-scale MAE**.

---

# 5. The important conclusion

Seeing:

```text
BMA MAE > AIFS MAE
```

does **not** by itself indicate a bug.

It means:

> Your current BMA objective is optimizing probabilistic fit, while you're judging its point forecast with MAE.

However, there is one thing I **would** investigate before presenting the model as superior:

### Compare against a proper baseline

You should compare:

```text
AIFS
GFS
BMA
```

on a genuinely unseen period using a proper probabilistic scoring rule as well as MAE.

And for your specific prototype, I would also inspect the learned parameters:

```text
location
AIFS weight
GFS weight
AIFS slope/intercept
GFS slope/intercept
AIFS sigma
GFS sigma
```

because a location where:

```text
AIFS MAE << GFS MAE
```

but BMA still gives substantial GFS weight is an obvious candidate for explaining the larger BMA MAE.

Your existing results already show this behavior in places such as New Delhi and Chennai.