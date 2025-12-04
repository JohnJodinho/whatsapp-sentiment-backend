import json
from typing import Dict, Any

def serialize_analytics(analytics_json: Dict[str, Any] | None) -> str:
    if not analytics_json:
        return "null"

    out = {}

    # ------------------------
    # GENERAL DASHBOARD DIGEST
    # ------------------------
    gd = analytics_json.get("general_dashboard")
    if gd:
        digest = {}

        # KPI summary (values only)
        kpi = gd.get("kpiMetrics") or []
        digest["kpi"] = {
            m["label"]: m.get("value")
            for m in kpi
        }

        # Contribution: reduce to top 3 names + percentages
        ctr = gd.get("contribution")
        if ctr:
            if ctr.get("type") == "multi":
                data = ctr["data"]
                top = sorted(data, key=lambda x: x["messages"], reverse=True)[:3]
                digest["top_contributors"] = [
                    {"name": t["name"], "messages": t["messages"]} for t in top
                ]

            elif ctr.get("type") == "two":
                a = ctr["data"]["participants"]
                digest["top_contributors"] = a

            elif ctr.get("type") == "single":
                digest["top_contributors"] = [ctr["data"]]

        # Activity summary (not raw arrays)
        act = gd.get("activity")
        if act:
            digest["activity_categories"] = act.get("labels")
            digest["activity_participants"] = [
                p["name"] for p in act.get("participants", [])
            ]

        # Timeline: compress to most recent summary only
        tl = gd.get("timeline")
        if tl:
            recent = tl[-1]   # last month is always enough
            digest["recent_month"] = {
                "month": recent.get("month"),
                "totalMessages": recent.get("totalMessages"),
                "peakDay": recent.get("peakDay"),
            }

        out["general"] = digest

    # ------------------------
    # SENTIMENT DASHBOARD DIGEST
    # ------------------------
    sd = analytics_json.get("sentiment_dashboard")
    if sd:
        sdig = {}

        # KPI
        if sd.get("kpiData"):
            k = sd["kpiData"]
            sdig["overall"] = {
                "score": k.get("overallScore"),
                "pos": k.get("positivePercent"),
                "neg": k.get("negativePercent"),
                "neu": k.get("neutralPercent"),
            }

        # Breakdown: top 3 participants only
        if sd.get("breakdownData"):
            b = sd["breakdownData"]
            top = sorted(b, key=lambda x: x["Positive"], reverse=True)[:3]
            sdig["top_sentiment"] = [
                {"name": x["name"], "pos": x["Positive"], "neg": x["Negative"]} 
                for x in top
            ]

        # Highlights: strip to short summaries
        hl = sd.get("highlightsData")
        if hl:
            def short(msg):
                text = msg.get("text") or ""
                return {
                    "sender": msg.get("sender"),
                    "score": msg.get("score"),
                    "snippet": text[:80],  # small, safe
                }
            sdig["highlights"] = {
                "pos": [short(m) for m in hl.get("topPositive", [])[:3]],
                "neg": [short(m) for m in hl.get("topNegative", [])[:3]],
            }

        out["sentiment"] = sdig

    return json.dumps(out, separators=(",", ":"))
