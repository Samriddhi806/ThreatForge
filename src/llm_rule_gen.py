# Phase 3: LLM-Driven Suricata Rule Generation
import anthropic
import idstools.rule
import hashlib
import json
import os
import pandas as pd
import joblib
import numpy as np
import time

print("=" * 55)
print("  Phase 3 — LLM Rule Generator (Claude API)")
print("=" * 55)

os.makedirs("generated_rules", exist_ok=True)
os.makedirs("logs",            exist_ok=True)

# MITRE ATT&CK mapping 
MITRE_MAP = {
    "DoS":            {"id": "T1499", "tactic": "Impact",
                       "desc": "Endpoint Denial of Service"},
    "DDoS":           {"id": "T1498", "tactic": "Impact",
                       "desc": "Network Denial of Service"},
    "Botnet":         {"id": "T1071", "tactic": "Command and Control",
                       "desc": "Application Layer Protocol C2"},
    "Bruteforce":     {"id": "T1110", "tactic": "Credential Access",
                       "desc": "Brute Force — password spraying"},
    "Portscan":       {"id": "T1046", "tactic": "Discovery",
                       "desc": "Network Service Discovery"},
    "WebAttacks":     {"id": "T1190", "tactic": "Initial Access",
                       "desc": "Exploit Public-Facing Application"},
    "Infiltration":   {"id": "T1041", "tactic": "Exfiltration",
                       "desc": "Exfiltration Over C2 Channel"},
    "Benign":         None,
    "Synthetic-Attack": {"id": "T1046", "tactic": "Discovery",
                          "desc": "Adversarially crafted evasion attempt"},
}

#  State 
client      = anthropic.Anthropic()
seen_hashes = set()
rule_log    = []
SID_START   = 9000001

def get_next_sid():
    existing = [f for f in os.listdir("generated_rules")
                if f.endswith(".rules")]
    return SID_START + len(existing)

def generate_rule(features: dict, attack_type: str,
                  sid: int, confidence: float) -> str | None:
    """Send alert context to Claude and get a Suricata rule back."""

    mitre = MITRE_MAP.get(attack_type,
            {"id": "T1046", "tactic": "Discovery", "desc": "Unknown"})
    if mitre is None:
        return None

    prompt = f"""You are a senior network security engineer writing Suricata IDS rules.

Detected attack: {attack_type}
Confidence score: {confidence:.2%}
MITRE ATT&CK technique: {mitre['id']} — {mitre['desc']} ({mitre['tactic']})

Flow statistics from the detected network connection:
  Protocol        : {features.get('Protocol', 'TCP')}
  Flow duration   : {features.get('Flow Duration', 0):.0f} microseconds
  Total fwd pkts  : {features.get('Total Fwd Packets', 0):.0f}
  Total bwd pkts  : {features.get('Total Backward Packets', 0):.0f}
  Fwd pkt length mean : {features.get('Fwd Packet Length Mean', 0):.1f} bytes
  Bwd pkt length mean : {features.get('Bwd Packet Length Mean', 0):.1f} bytes
  Packet length var   : {features.get('Packet Length Variance', 0):.1f}
  Flow bytes/s    : {features.get('Flow Bytes/s', 0):.1f}
  FIN flag count  : {features.get('FIN Flag Count', 0):.0f}
  PSH flag count  : {features.get('PSH Flag Count', 0):.0f}
  SYN flag count  : {features.get('SYN Flag Count', 0):.0f}
  Flow IAT max    : {features.get('Flow IAT Max', 0):.1f}
  Down/Up ratio   : {features.get('Down/Up Ratio', 0):.2f}

Write exactly ONE valid Suricata 7.x IDS rule to detect this attack pattern.
Use SID {sid}.
Output the rule only — no explanation, no markdown, no extra text.
Format: alert <proto> any any -> any any (msg:"ARIA - {attack_type} - {mitre['id']}"; <options>; sid:{sid}; rev:1;)"""

    try:
        response = client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 300,
            messages   = [{"role": "user", "content": prompt}]
        )
        rule_str = response.content[0].text.strip()

        # Remove any accidental markdown fences
        rule_str = rule_str.replace("```", "").strip()

        return rule_str

    except Exception as e:
        print(f"      [!] API error: {e}")
        return None

def validate_rule(rule_str: str) -> bool:
    """Check rule syntax using idstools parser."""
    try:
        parsed = idstools.rule.parse(rule_str)
        return parsed is not None
    except Exception:
        return False

def save_rule(rule_str: str, attack_type: str, sid: int) -> str | None:
    """Deduplicate and save rule to generated_rules/"""
    h = hashlib.md5(rule_str.encode()).hexdigest()[:10]
    if h in seen_hashes:
        print(f"      [=] Duplicate rule skipped (hash {h})")
        return None
    seen_hashes.add(h)

    fname = f"{attack_type}_{sid}_{h}.rules"
    path  = os.path.join("generated_rules", fname)
    with open(path, "w") as f:
        f.write(rule_str + "\n")

    return path

def process_alert(row: dict, attack_type: str,
                  confidence: float, sid: int) -> dict:
    """Full pipeline: features → LLM → validate → save"""
    print(f"\n  Alert: {attack_type} (confidence {confidence:.1%})")
    print(f"  SID  : {sid}")

    rule_str = generate_rule(row, attack_type, sid, confidence)

    if rule_str is None:
        print(f"      [!] Rule generation failed")
        return {"attack": attack_type, "status": "api_error", "rule": None}

    print(f"      Generated: {rule_str[:80]}...")

    if not validate_rule(rule_str):
        print(f"      [!] Rule failed syntax validation — skipped")
        return {"attack": attack_type, "status": "invalid_syntax",
                "rule": rule_str}

    path = save_rule(rule_str, attack_type, sid)

    if path is None:
        return {"attack": attack_type, "status": "duplicate", "rule": rule_str}

    print(f"      [+] Saved → {path}")
    return {
        "attack":     attack_type,
        "status":     "success",
        "rule":       rule_str,
        "sid":        sid,
        "saved_to":   path,
    }

#  Main: test against real detections from test set
print("\n[1/4] Loading model and test data...")

model        = joblib.load("models/model_final.pkl")
test_df      = pd.read_parquet("data/processed/test.parquet")
feature_cols = [c for c in test_df.columns
                if c not in ['binary_label', 'attack_type']]

X_test  = test_df[feature_cols].values
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Find real attack detections
detected_mask = (y_pred == 1) & (test_df['binary_label'].values == 1)
detected_df   = test_df[detected_mask].copy()
detected_df['confidence'] = y_proba[detected_mask]

print(f"      Total detections : {detected_mask.sum():,}")
print(f"      Test set attacks : {(test_df['binary_label']==1).sum():,}")

# 2. Generate one rule per attack class 
print("\n[2/4] Generating one rule per attack class...")
print("      (API calls take 2–5 seconds each)\n")

results = []
sid     = get_next_sid()

for attack_type in sorted(detected_df['attack_type'].unique()):
    if attack_type == 'Benign':
        continue
    if MITRE_MAP.get(attack_type) is None:
        continue

    # Pick the highest-confidence detection for this class
    subset = detected_df[detected_df['attack_type'] == attack_type]
    if len(subset) == 0:
        continue

    best_row = subset.loc[subset['confidence'].idxmax()]
    features = dict(zip(feature_cols, best_row[feature_cols].values))
    conf     = float(best_row['confidence'])

    result = process_alert(features, attack_type, conf, sid)
    result['attack_type'] = attack_type
    results.append(result)
    sid += 1

    time.sleep(1)  # avoid rate limiting

# 3. Save results log
print("\n[3/4] Saving results log...")

with open("logs/llm_rule_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("      Saved → logs/llm_rule_results.json")

# 4. Summary
print("\n[4/4]  LLM Rule Generation complete!")

success = [r for r in results if r['status'] == 'success']
failed  = [r for r in results if r['status'] != 'success']

print(f"""

  ARIA — LLM Rule Generation Summary                  
-----------------------------------------------------------
  Attack classes processed : {len(results)}                        
  Rules generated OK       : {len(success)}                        
  Failed / duplicate       : {len(failed)}                        

""")

print("  Generated rules:")
for r in success:
    print(f"    {r['attack_type']:15s} → {r.get('saved_to','')}")

if failed:
    print("\n  Failed:")
    for r in failed:
        print(f"    {r['attack_type']:15s} → {r['status']}")

print("\n  Generated rules saved in: generated_rules/")
print("  Next: python src/dashboard.py")

# Bonus: print all generated rules
print("\n" + "="*55)
print("  Generated Suricata rules:")
print("="*55)
for r in success:
    print(f"\n  # {r['attack_type']} — MITRE {MITRE_MAP[r['attack_type']]['id']}")
    print(f"  {r['rule']}")