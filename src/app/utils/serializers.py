import json
from typing import Dict, Any

def serialize_analytics(analytics_json: Dict[str, Any] | None) -> str:
    """
    Intelligently summarizes the analytics JSON to maximize 
    informational density for the LLM while minimizing token cost.
    """
    if not analytics_json:
        return "null"
    
    clean_data = {}

    # --- 1. Process General Dashboard ---
    # Based on dashboardData.ts structure
    if "general_dashboard" in analytics_json:
        gd = analytics_json["general_dashboard"]
        clean_gd = {}
        
        # A. KPIs: Keep label/value, REMOVE sparklines (visual noise)
        if "kpiMetrics" in gd:
            clean_gd["kpiMetrics"] = [
                {k: v for k, v in m.items() if k != "sparkline"} 
                for m in gd.get("kpiMetrics", [])
            ]
            
        # B. Contribution: Keep fully (Answer "Who sent the most?")
        if "contribution" in gd:
            clean_gd["contribution"] = gd["contribution"]
            
        # C. Activity: Keep fully (Answer "Who sends links/media?")
        if "activity" in gd:
            clean_gd["activity"] = gd["activity"]
            
        # D. Timeline: High value, but limit history (last 12 months)
        #    Removes 'messagesOverTime' as this table provides better summary
        if "timeline" in gd:
            # Only keep essential fields from ChatSegmentBase/Multi/Two
            clean_gd["timeline_summary"] = [
                {
                    k: v for k, v in item.items() 
                    if k in ["month", "totalMessages", "peakDay", "activeParticipants", "mostActive", "conversationBalance"]
                }
                for item in gd.get("timeline", [])[:12] # Top 12 most recent months
            ]
            
        # E. Heatmaps: Keep compact (Answer "When are they active?")
        if "activityByDay" in gd:
             # Remove 'fill' color code, keep day/messages
             clean_gd["activityByDay"] = [
                 {k: v for k, v in d.items() if k != "fill"}
                 for d in gd.get("activityByDay", [])
             ]
             
        if "hourlyActivity" in gd:
             clean_gd["hourlyActivity"] = gd.get("hourlyActivity")

        clean_data["general"] = clean_gd

    # --- 2. Process Sentiment Dashboard ---
    # Based on sentimentDashboardData.ts structure
    if "sentiment_dashboard" in analytics_json:
        sd = analytics_json["sentiment_dashboard"]
        clean_sd = {}
        
        # A. KPIs: Keep fully
        if "kpiData" in sd:
            clean_sd["kpiData"] = sd["kpiData"]
            
        # B. Breakdown: Keep fully (Answer "Who is most positive?")
        if "breakdownData" in sd:
            clean_sd["breakdownData"] = sd["breakdownData"]
            
        # C. Highlights: CRITICAL (Answer "Why is it negative?")
        if "highlightsData" in sd:
            clean_sd["highlights"] = sd["highlightsData"]
            
        # D. Time/Trend: Remove 'trendData' (too verbose). 
        #    Keep day/hour aggregates as they are smaller.
        if "dayData" in sd:
            clean_sd["dayData"] = sd["dayData"]
        
        #    Summarize hourly to just top 3 peaks to save tokens?
        #    Or keep as is (24 items is manageable). Let's keep as is.
        if "hourData" in sd:
            clean_sd["hourData"] = sd["hourData"]

        clean_data["sentiment"] = clean_sd

    # Return the slimmed-down version
    # separators=(",", ":") removes all whitespace to save tokens
    return json.dumps(clean_data, separators=(",", ":"))