

from flask import Flask, request, jsonify, render_template_string
import joblib, json, numpy as np, os

app = Flask(__name__)

# ── Load model & encoders ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
with open(os.path.join(BASE_DIR, "encoders.json")) as f:
    ENCODERS = json.load(f)

THRESHOLD = 0.47   # ← change to 0.55 or 0.60 for stricter screening

# ── HTML Template ──────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Credit Risk Checker</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 36px 40px;
    width: 100%;
    max-width: 680px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.4);
  }

  .header { text-align: center; margin-bottom: 32px; }
  .header .icon { font-size: 2.6rem; margin-bottom: 8px; }
  .header h1 { color: #f1f5f9; font-size: 1.6rem; font-weight: 700; }
  .header p  { color: #94a3b8; font-size: 0.875rem; margin-top: 6px; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field.full { grid-column: span 2; }

  label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  input, select {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #f1f5f9;
    padding: 10px 14px;
    font-size: 0.95rem;
    outline: none;
    transition: border-color 0.2s;
    width: 100%;
  }
  input:focus, select:focus { border-color: #6366f1; }
  select option { background: #1e293b; }

  .submit-btn {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 10px;
    color: #fff;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.03em;
    transition: opacity 0.2s, transform 0.1s;
  }
  .submit-btn:hover   { opacity: 0.9; }
  .submit-btn:active  { transform: scale(0.99); }
  .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Result card */
  #result { margin-top: 24px; display: none; }

  .result-box {
    border-radius: 12px;
    padding: 24px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
  }
  .result-box.good { background: #052e16; border: 1px solid #166534; }
  .result-box.bad  { background: #450a0a; border: 1px solid #991b1b; }
  .result-box.loading { background: #1e1b4b; border: 1px solid #4338ca; }

  .result-icon { font-size: 2.8rem; flex-shrink: 0; }
  .result-text h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 4px; }
  .result-box.good .result-text h2 { color: #4ade80; }
  .result-box.bad  .result-text h2 { color: #f87171; }
  .result-box.loading .result-text h2 { color: #a5b4fc; }
  .result-text p  { font-size: 0.875rem; color: #94a3b8; line-height: 1.5; }

  /* Gauge bar */
  .gauge-wrap { margin-top: 18px; }
  .gauge-label { display: flex; justify-content: space-between;
                  font-size: 0.78rem; color: #64748b; margin-bottom: 6px; }
  .gauge-track {
    height: 10px; background: #0f172a;
    border-radius: 99px; overflow: hidden;
  }
  .gauge-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.8s ease;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
  }

  /* Score badges */
  .badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
  .badge {
    padding: 4px 12px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 600;
  }
  .badge-blue   { background: #1e3a5f; color: #93c5fd; }
  .badge-purple { background: #2e1065; color: #c4b5fd; }

  .divider {
    height: 1px; background: #334155; margin: 24px 0;
  }

  .threshold-note {
    text-align: center; font-size: 0.78rem; color: #475569; margin-top: 16px;
  }

  @media (max-width: 520px) {
    .grid { grid-template-columns: 1fr; }
    .field.full { grid-column: span 1; }
    .card { padding: 24px 20px; }
  }
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="icon">🏦</div>
    <h1>Credit Risk Checker</h1>
    <p>Enter customer details to assess creditworthiness using our AI model</p>
  </div>

  <form id="form">
    <div class="grid">

      <div class="field">
        <label>Age</label>
        <input type="number" name="age" min="18" max="100" placeholder="e.g. 35" required/>
      </div>

      <div class="field">
        <label>Sex</label>
        <select name="sex" required>
          <option value="" disabled selected>Select</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
      </div>

      <div class="field">
        <label>Job Level</label>
        <select name="job" required>
          <option value="" disabled selected>Select</option>
          <option value="0">0 — Unskilled (non-resident)</option>
          <option value="1">1 — Unskilled (resident)</option>
          <option value="2">2 — Skilled</option>
          <option value="3">3 — Highly Skilled</option>
        </select>
      </div>

      <div class="field">
        <label>Housing</label>
        <select name="housing" required>
          <option value="" disabled selected>Select</option>
          <option value="own">Own</option>
          <option value="rent">Rent</option>
          <option value="free">Free</option>
        </select>
      </div>

      <div class="field">
        <label>Saving Account</label>
        <select name="saving_accounts" required>
          <option value="" disabled selected>Select</option>
          <option value="little">Little</option>
          <option value="moderate">Moderate</option>
          <option value="quite rich">Quite Rich</option>
          <option value="rich">Rich</option>
          <option value="unknown">Unknown / None</option>
        </select>
      </div>

      <div class="field">
        <label>Checking Account</label>
        <select name="checking_account" required>
          <option value="" disabled selected>Select</option>
          <option value="little">Little</option>
          <option value="moderate">Moderate</option>
          <option value="rich">Rich</option>
          <option value="unknown">Unknown / None</option>
        </select>
      </div>

      <div class="field">
        <label>Credit Amount (€)</label>
        <input type="number" name="credit_amount" min="100" max="100000"
               placeholder="e.g. 5000" required/>
      </div>

      <div class="field">
        <label>Duration (months)</label>
        <input type="number" name="duration" min="1" max="120"
               placeholder="e.g. 24" required/>
      </div>

      <div class="field full">
        <label>Purpose</label>
        <select name="purpose" required>
          <option value="" disabled selected>Select purpose of credit</option>
          <option value="car">Car</option>
          <option value="furniture/equipment">Furniture / Equipment</option>
          <option value="radio/TV">Radio / TV</option>
          <option value="domestic appliances">Domestic Appliances</option>
          <option value="repairs">Repairs</option>
          <option value="education">Education</option>
          <option value="business">Business</option>
          <option value="vacation/others">Vacation / Others</option>
        </select>
      </div>

    </div><!-- end grid -->

    <button class="submit-btn" type="submit" id="submitBtn">
      Assess Credit Risk
    </button>
  </form>

  <div id="result">
    <div class="divider"></div>
    <div id="resultBox" class="result-box loading">
      <div class="result-icon" id="resultIcon">⏳</div>
      <div class="result-text">
        <h2 id="resultTitle">Analysing…</h2>
        <p  id="resultDesc"></p>
        <div class="badges" id="badges"></div>
      </div>
    </div>
    <div class="gauge-wrap" id="gaugeWrap" style="display:none">
      <div class="gauge-label">
        <span>High Risk</span><span id="gaugeLabel"></span><span>Low Risk</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" id="gaugeFill" style="width:0%"></div>
      </div>
    </div>
  </div>

  <p class="threshold-note">Model: Random Forest + SMOTE &nbsp;|&nbsp; Threshold: {{ threshold }}</p>
</div>

<script>
document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn    = document.getElementById('submitBtn');
  const result = document.getElementById('result');
  const box    = document.getElementById('resultBox');
  const icon   = document.getElementById('resultIcon');
  const title  = document.getElementById('resultTitle');
  const desc   = document.getElementById('resultDesc');
  const badges = document.getElementById('badges');
  const gauge  = document.getElementById('gaugeWrap');
  const fill   = document.getElementById('gaugeFill');
  const glabel = document.getElementById('gaugeLabel');

  btn.disabled = true;
  btn.textContent = 'Analysing…';

  // Show loading state
  result.style.display = 'block';
  box.className = 'result-box loading';
  icon.textContent = '⏳';
  title.textContent = 'Running model…';
  desc.textContent  = '';
  badges.innerHTML  = '';
  gauge.style.display = 'none';

  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();

    if (data.error) {
      box.className    = 'result-box bad';
      icon.textContent = '❌';
      title.textContent = 'Error';
      desc.textContent  = data.error;
    } else {
      const isGood = data.prediction === 'Good';
      box.className    = isGood ? 'result-box good' : 'result-box bad';
      icon.textContent = isGood ? '✅' : '⚠️';
      title.textContent = isGood ? 'Customer is SAFE — Good Credit' : 'Customer is RISKY — Bad Credit';
      desc.textContent  = isGood
        ? 'This applicant is likely to repay the loan. Credit can be approved.'
        : 'This applicant has a high risk of default. Review before approving.';

      badges.innerHTML = `
        <span class="badge badge-blue">Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
        <span class="badge badge-purple">P(Good): ${(data.prob_good * 100).toFixed(1)}%</span>
      `;

      gauge.style.display = 'block';
      const pct = (data.prob_good * 100).toFixed(1);
      fill.style.width = pct + '%';
      glabel.textContent = pct + '% Good';
    }
  } catch (err) {
    box.className    = 'result-box bad';
    icon.textContent = '❌';
    title.textContent = 'Connection Error';
    desc.textContent  = 'Could not reach the server. Make sure app.py is running.';
  }

  btn.disabled = false;
  btn.textContent = 'Assess Credit Risk';
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, threshold=THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Map saving/checking "unknown" → "nan" (matches training encoding) ──
        saving   = data.get("saving_accounts", "unknown")
        checking = data.get("checking_account", "unknown")
        if saving   == "unknown": saving   = "nan"
        if checking == "unknown": checking = "nan"

        # ── Encode categoricals ──
        def encode(col, val):
            mapping = ENCODERS[col]
            if val not in mapping:
                raise ValueError(f"Unknown value '{val}' for '{col}'")
            return mapping[val]

        features = np.array([[
            int(data["age"]),
            encode("Sex",              data["sex"]),
            int(data["job"]),
            encode("Housing",          data["housing"]),
            encode("Saving accounts",  saving),
            encode("Checking account", checking),
            int(data["credit_amount"]),
            int(data["duration"]),
            encode("Purpose",          data["purpose"]),
        ]])

        prob_good  = float(model.predict_proba(features)[0][1])
        prediction = "Good" if prob_good >= THRESHOLD else "Bad"
        confidence = prob_good if prediction == "Good" else (1 - prob_good)

        return jsonify({
            "prediction": prediction,
            "prob_good":  round(prob_good, 4),
            "confidence": round(confidence, 4),
            "threshold":  THRESHOLD,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏦  Credit Scoring App")
    print(f"   Model     : Random Forest + SMOTE")
    print(f"   Threshold : {THRESHOLD}")
    print(f"   Running   : http://localhost:5000\n")
    app.run(debug=True, port=5000)
