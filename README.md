## Abhi ye karo ek ek karke

---

## Step 1 — Sab files commit karo

```bash
git add .
git commit -m "feat: langsmith integration + async redis + qdrant pool fix"
git push origin main
```

---

## Step 2 — Local test karo docker-compose se

Redis locally chahiye pehle:

```bash
docker-compose up -d redis
```

Phir services start karo:

```bash
docker-compose up --build
```

---

## Step 3 — Test karo locally

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"query": "swiggy delivery issues", "company": "swiggy"}'
```

PowerShell me:

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/analyze" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"query": "swiggy delivery issues", "company": "swiggy"}'
```

---

## Step 4 — LangSmith dashboard check karo

**https://smith.langchain.com** pe jao — `feedbacklens-v2` project me traces dikhne chahiye.

---

Karo ye aur output batao — error aaye ya success dono cases me. Phir ECR/EKS shuru karte hain.