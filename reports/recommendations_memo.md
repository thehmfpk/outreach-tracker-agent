# Recommendations Memo
**Project:** Predictive Dashboard — Bank Customer Service Centre
**Prepared by:** SafeX Solutions — AI & ML Department, Week 4 Sprint
**Date:** Week 4

---

## Summary

Using 18 months of (simulated) daily operational data — call volume, wait
times, branch foot traffic, satisfaction (CSAT), and social engagement — we
built a simple, explainable forecasting model (linear regression + moving
average) to project the next 30 days of activity. Three recommendations
follow directly from that forecast.

## 1. Adjust staffing ahead of forecasted call-volume peaks
Forecasted call volume shows a repeating weekly pattern, with Monday and
Friday consistently highest and weekends consistently lowest. Shifting 1–2
additional agents onto Monday/Friday shifts (and reducing weekend coverage
accordingly) should reduce peak-day wait times without increasing total
headcount hours.

## 2. Invest in self-service / chatbot deflection for routine queries
Wait time and CSAT move inversely in the data: as average wait time rises,
satisfaction falls. Deflecting simple, high-volume requests (balance
checks, statement requests, branch hours) to a self-service channel or
chatbot should lower average wait time and lift CSAT, independent of
staffing changes.

## 3. Time marketing/social pushes to forecasted engagement dips
Social engagement follows a roughly monthly cycle with predictable dips.
Scheduling promotional or educational content just *before* a forecasted
dip (rather than reacting after engagement has already dropped) should
smooth engagement and reduce the size of the troughs.

---

*Note: this memo is generated from simulated sample data for demonstration
purposes. Recommendations should be re-validated once real operational
data is connected.*
