## HPA Decision

Bhai pehle HPA clear karte hain:

**Old project me kya tha:** Sirf gateway aur orchestrator pe HPA tha — understanding aur recommendation agent pe nahi tha. Ye correct approach hai.

**Is project me maine sab pe HPA diya** — ye galat tha, tune sahi pakda.

**Production standard kya hai:**

```
Gateway      → HPA ✅ (public traffic face karta hai)
Orchestrator → HPA ✅ (gateway ke requests handle karta hai)
Insight agent → HPA ✅ (heavy operation — Qdrant + LLM)
Understanding → ❌ NO HPA (lightweight, fast, rarely bottleneck)
Recommendation → ❌ NO HPA (lightweight, fast, rarely bottleneck)
```

Toh **3 HPA files rakho, 2 delete karo:**

```powershell
# Delete these 2
del infrastructure/kubernetes/understanding-agent/hpa.yaml
del infrastructure/kubernetes/recommendation-agent/hpa.yaml
```

Keep karo:
```
gateway/hpa.yaml          ✅
orchestrator/hpa.yaml     ✅
insight-agent/hpa.yaml    ✅
```

---

## ragas_eval.py Decision

**Haan — delete karo.** Ab `langsmith_eval.py` me RAGAS fully integrated hai. Purana file redundant hai.

```powershell
del ragas_eval.py
```

---

## Commit karo

```powershell
git add .
git commit -m "chore: remove redundant ragas_eval.py and unnecessary HPA files"
git push origin main
```

---

## EKS ka kya status hai?

```powershell
eksctl get cluster --region us-east-1
```

Paste karo — ready hoga toh deployment shuru karte hain.