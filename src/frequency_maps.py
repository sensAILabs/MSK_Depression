import pandas as pd


F1 = {
    # ---- daily ----
    "OD": 1,
    "QD": 1,
    "DAILY": 1,
    "QDAY": 1,
    "AM": 1,
    "PM": 1,
    "HS": 1,
    "QHS": 1,
    "QAM": 1,
    "ID": 1,
    "ONE TABLET DAILY": 1,

    # ---- twice daily ----
    "BID": 0.5,
    "Q12H": 0.5,

    # ---- three times daily ----
    "TID": 1 / 3,
    "Q8H": 1 / 3,

    # ---- four times daily ----
    "QID": 0.25,
    "Q6H": 0.25,

    # ---- very frequent ----
    "Q4H": 1 / 6,

    # ---- other intervals ----
    "QOD": 2,          # every other day
    "WKLY": 7,
    "WEEKLY": 7,
    "MONTHLY": 30,

    # ---- ambiguous / unscheduled ----
    "Q4H-Q6H": pd.NA,
    "PRN": pd.NA,
}

F2 = {
    # ---- once daily ----
    "1 TAB QD 1 MO28": 1,
    "1 TAB QD 3 MO28": 1,
    "1 TAB QD 10 DAY": 1,
    "0.5 TAB QD 3 MO30": 1,
    "0.5 TAB QD 1 MO30": 1,
    "1 TAB QAM 1 MO30": 1,
    "1 TAB QPM 1 MO30": 1,
    "1 TAB QHS 3 MO30": 1,
    "1 TAB QHS 60 DAY": 1,

    # ---- twice daily ----
    "1 TAB BID 100 DAY": 0.5,
    "1 TAB BID 90 DAY": 0.5,
    "1 TAB BID 5 DAY": 0.5,
    "1 APPLN BID 30 DAY": 0.5,

    # ---- three times daily ----
    "1 TAB TID 1 MO30": 1 / 3,
    "1 TAB TID 30 DAY": 1 / 3,
    "1 TAB TID 10 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 10 DAY": 0.25,

    # ---- explicit daily count ----
    "5 X PER DAY": 1 / 5,
}

F3 = {
    # ---- once daily ----
    "1 TAB QAM 30 DAY": 1,
    "QPM": 1,
    "1 PO OD": 1,
    "1 TAB QD 7 DAY": 1,
    "2 TAB QD 30 DAY": 1,
    "2 TAB QD 1 MO30": 1,
    "0.5 TAB QD 30 DAY": 1,
    "0.5 TAB QD 60 DAY": 1,
    "0.5 TAB QD 100 DAY": 1,
    "1 TAB QHS 100 DAY": 1,

    # ---- twice daily ----
    "1 APPLN BID 1 MO30": 0.5,
    "1 CAP BID 1 MO30": 0.5,
    "1 CAP BID 30 DAY": 0.5,

    # ---- three times daily ----
    "5 ML TID 10 DAY": 1 / 3,
    "1 CAP TID 7 DAY": 1 / 3,
    "1 TAB TID 7 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 7 DAY": 0.25,

    # ---- explicit daily count ----
    "5X/DAY": 1 / 5,
}

F4 = {
    # ---- once daily ----
    "ONE TABLET ONCE DAILY": 1,
    "ONE CAPSULE DAILY": 1,
    "ONCE": 1,
    "1 TAB QD 21 DAY": 1,
    "0.5 TAB QD 90 DAY": 1,
    "2 TAB QD 3 MO30": 1,
    "1 TAB QHS 90 DAY": 1,   

    # ---- twice daily ----
    "1 APPLN BID 10 DAY": 0.5,
    "1 TAB BID 14 DAY": 0.5,
    "2 TAB BID 30 DAY": 0.5,

    # ---- hour-based ----
    "Q2H": 1 / 12,   # every 2 hours
    "Q3H": 1 / 8,    # every 3 hours
}

F5 = {
    # ---- once daily ----
    "ONE DAILY": 1,
    "TAKE ONE TABLET DAILY": 1,
    "1 TAB QD 5 DAY": 1,
    "1 TAB QD 4 WEEK": 1,
    "1 TAB QD 2 YEAR": 1,

    # ---- twice daily ----
    "2 TAB BID 1 MO30": 0.5,
    "1 TAB BID 3 DAY": 0.5,

    # ---- three times daily ----
    "1 TAB TID 3 MO30": 1 / 3,

    # ---- long interval ----
    "Q72H": 3,   # every 72 hours
}

F6 = {
    # ---- every N days ----
    "Q2DAY": 2,
    "Q3DAY": 3,
    "ALT. DAYS": 2,

    # ---- weekly / monthly ----
    "QWEEKLY": 7,
    "2 X PER WEEK": 3.5,
    "3 X PER WEEK": 7 / 3,
    "BIWEEKLY": 14,
    "QMONTHLY": 30,
    "Q3MONTHS": 90,

    # ---- daily (timing / meals, still once per day) ----
    "WITH SUPPER": 1,
    "WITH LUNCH": 1,
    "WITH BREAKFAST": 1,
    "AC MEALS": 1,
    "AC BREAKFAST": 1,
    "1 DAILY": 1,

    # ---- stat / one-time ----
    "STAT": pd.NA,
}

F7 = {
    # ---- once daily (duration info ignored) ----
    "AC SUPPER": 1,
    "0.5 TAB QD 2 MO30": 1,
    "1 TAB QD 6 MO30": 1,
    "1 TAB QD 14 DAY": 1,
    "1 TAB QD 2 WEEK": 1,
    "1 UNIT ONCE 1 DAY": 1,   # 语义上就是 once

    # ---- twice daily ----
    "1 PUFF BID 1 MO30": 0.5,
    "2 TAB BID 3 MO30": 0.5,
    "1 CAP BID 7 DAY": 0.5,
    "1 TAB BID 2 MO30": 0.5,
    "0.5 TAB BID 30 DAY": 0.5,
    "1 TAB BID 28 DAY": 0.5,
    "1 TAB BID 2 WEEK": 0.5,

    # ---- four times daily ----
    "1 CAP QID 10 DAY": 0.25,
}

F8 = {
    # ---- once daily ----
    "1 CAP QD 2 MO30": 1,
    "ONE OD": 1,
    "1.5 TAB QD 30 DAY": 1,

    # ---- twice daily ----
    "1 TAB BID 15 DAY": 0.5,
    "0.5 TAB BID 1 MO30": 0.5,

    # ---- three times daily ----
    "1 TAB TID 100 DAY": 1 / 3,

    # ---- bedtime (once daily) ----
    "1 TO 2 TAB QHS 30 DAY": 1,
    "2 TAB QHS 1 MO30": 1,

    # ---- biweekly ----
    "Q2WEEK": 14,
}

F9 = {
    # ---- once daily ----
    "1 CAP QAM 1 MO30": 1,
    "1 CAP QHS 30 DAY": 1,
    "1 CAP QHS 1 MO30": 1,
    "1 TAB QHS 2 MO30": 1,
    "2 TAB QHS 30 DAY": 1,
    "2 TAB QHS 30 DAY": 1,  # 同 key 重复不影响，保留一条即可
    "1 TO 2 TAB QD 30 DAY": 1,
    "1 APPLN QD 30 DAY": 1,
    "ONE TABLET DAILY EACH MORNING": 1,
    "PO OD": 1,
    "1 CAP QD 28 DAY": 1,
    "1.5 TAB QD 1 MO30": 1,
    "2 TAB QD 60 DAY": 1,
    "2 TAB QD 90 DAY": 1,
    "1 TAB ONCE 3 MO30": 1,
    "1 TAB QD 30 DAY": 1,  # 如果你数据里确实出现（你上面没列，但常见）

    # ---- twice daily ----
    "1 PUFF BID 30 DAY": 0.5,
    "2 PUFF BID 30 DAY": 0.5,
    "2 TAB BID 60 DAY": 0.5,
    "1 CAP BID 10 DAY": 0.5,
    "1 APPLN BID 14 DAY": 0.5,
    "1 TAB BID 1 MO28": 0.5,
    "1 CAP BID 3 MO30": 0.5,

    # ---- weekly ----
    "1 TAB 1X/WEEK 3 MO30": 7,

    # ---- every 5 minutes ----
    "Q5MIN": 5 / 1440,  # 5 minutes = 5/1440 days
}

F10 = {
    "1 QHS": 1,
    "1 TAB QPM 3 MO30": 1,
    "1 TAB QD 4 MO30": 1,
    "2 TAB QD 100 DAY": 1,
    "2 SPRAY QD 30 DAY": 1,
    "2 SPRAY QD 1 MO30": 1,
    "1 TAB QHS 28 DAY": 1,
    "2 CAP QD 30 DAY": 1,
    "1 TAB QAM 4 WEEK": 1,

    "1 APPLN BID 7 DAY": 0.5,
    "0.5 TAB BID 100 DAY": 0.5,
    "0.5 TAB BID 3 MO30": 0.5,
    "1 INHLN BID 30 DAY": 0.5,

    "1 TAB TID 90 DAY": 1 / 3,
    "5 ML TID 7 DAY": 1 / 3,
    "1 CAP TID 1 MO30": 1 / 3,

    "1 CAP QID 7 DAY": 0.25,
}

F11 = {
    # ---- once daily ----
    "1 TAB QAM 3 MO30": 1,
    "1 TAB QPM 2 MO30": 1,
    "1 ML ONCE 1 DAY": 1,

    # ---- twice daily ----
    "2 TAB BID 100 DAY": 0.5,
    "1 TAB BID 4 WEEK": 0.5,
    "1 APPLN BID 0": 0.5,
    "1 APPLN BID 2 WEEK": 0.5,

    # ---- three times daily ----
    "1 CAP TID 30 DAY": 1 / 3,
    "1 TAB TID 14 DAY": 1 / 3,
    "1 TAB TID 60 DAY": 1 / 3,
    "1 APPLN TID 10 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 1 MO30": 0.25,

    # ---- bedtime (once daily) ----
    "0.5 TO 1 TAB QHS 1 MO30": 1,
}

F12 = {
    # ---- once daily ----
    "WITH MEALS": 3,
    "1 APPLN QD 1 MO30": 1,
    "1 SPRAY QD 1 MO30": 1,
    "1 TAB QD 1 YEAR": 1,

    # ---- twice daily ----
    "1 CAP BID 60 DAY": 0.5,
    "1 TAB BID 1 WEEK": 0.5,
    "2 TAB BID 90 DAY": 0.5,
    "1 INHLN BID 1 MO30": 0.5,
    "2 PUFF BID 1 MO30": 0.5,

    # ---- four times daily ----
    "1 TAB QID 30 DAY": 0.25,
}

F13 = {
    # ---- once daily ----
    "1 TAB QD 200 DAY": 1,
    "ONE TABLET OD": 1,
    "1.5 TAB QD 3 MO30": 1,

    # ---- twice daily ----
    "0.5 TAB BID 60 DAY": 0.5,
    "0.5 TAB BID 90 DAY": 0.5,
    "2 TAB BID 10 DAY": 0.5,
}

F14 = {
    # ---- once daily ----
    "TAKE ONE TABLET TWICE DAILY": 0.5,
    "ONE TABLET PO DAILY": 1,
    "ONE TABLET BY MOUTH DAILY": 1,
    "PC SUPPER": 1,
    "AC LUNCH": 1,
    "1 TAB QD 3 DAY": 1,
    "1 PATCH QD 30 DAY": 1,
    "1 TO 1 TAB QD 30 DAY": 1,
    "2 TAB QHS 3 MO30": 1,
    "1 APPLN QHS 30 DAY": 1,
    "1.5 TAB QD 60 DAY": 1,
    "1 PO QD": 1,
    "0.5 TAB QPM 1 MO30": 1,
    "1 TAB QAM 100 DAY": 1,
    "1 CAP QD 1 DAY": 1,
    "2 CAP QD 1 MO30": 1,
    "1 TAB QHS 1 MO28": 1,

    # ---- twice daily ----
    "1 BID": 0.5,
    "1 CAP BID 5 DAY": 0.5,
    "5 ML BID 10 DAY": 0.5,
    "1 PUFF BID 3 MO30": 0.5,
    "1 PUFF BID 0": 0.5,
    "1 TO 2 PUFF BID 30 DAY": 0.5,
    "1 CAP BID 100 DAY": 0.5,

    # ---- three times daily ----
    "1 TAB TID 2 WEEK": 1 / 3,
    "1 TAB TID 5 DAY": 1 / 3,

    # ---- four times daily ----
    "2 PUFF QID 30 DAY": 0.25,

    # ---- weekly / monthly ----
    "1 TAB 1X/WEEK 1 MO30": 7,
    "1 TAB 1X/WEEK 4 WEEK": 7,
    "1 TAB 1X/WEEK 12 WEEK": 7,
    "Q3WEEKLY": 21,
    "1 ML 1X/MONTH 10 MO30": 30,
}

F15 = {
    # ---- hour-based (fixed) ----
    "Q1H": 1 / 24,  # every 1 hour

    # ---- once daily ----
    "2 TAB QD 7 DAY": 1,
    "1 CAP QAM 30 DAY": 1,
    "1 TAB QD 1 WEEK": 1,
    "ONE INHALATION DAILY": 1,
    "1 TO 1 TAB QD 1 MO30": 1,
    "1 TAB QHS 4 WEEK": 1,
    "1 TAB QAM 60 DAY": 1,
    "0.5 TO 1 TAB QHS 30 DAY": 1,
    "0.5 TAB QHS 1 MO30": 1,
    "3 TAB QD 1 MO30": 1,
    "1 TAB QD 12 MO30": 1,
    "1 APPLN QD 3 MO30": 1,
    "1 TAB QPM 30 DAY": 1,
    "1 TAB QD 15 DAY": 1,
    "1 CAP QPM 1 MO30": 1,

    # ---- twice daily ----
    "ONE TABLET TWICE DAILY": 0.5,
    "1 CAP BID 90 DAY": 0.5,
    "ONE BID": 0.5,
    "1 TAB BID 2 YEAR": 0.5,

    # ---- three times daily ----
    "1 CAP TID 14 DAY": 1 / 3,

    # ---- weekly ----
    "1 TAB 1X/WEEK 30 DAY": 7,

    # ---- four times daily (fixed, non-PRN) ----
    "1 TO 2 TAB QID 30 DAY": 0.25,
}

F16 = {
    # ---------- 固定小时 / 天 ----------
    "Q48H": 2,            # every 48 hours
    "Q2D": 2,             # every 2 days
    "Q2MONTHS": 60,       # approx 2 * 30 days

    # ---------- 固定每日 ----------
    "0.5 TAB QAM 1 MO30": 1,
    "1 TAB QAM 2 MO30": 1,
    "2 TAB QD 10 DAY": 1,
    "0.5 TAB QHS 30 DAY": 1,
    "1 TO 2 TAB QD 1 MO30": 1,
    "3 TAB QD 30 DAY": 1,

    # ---------- 固定 BID ----------
    "1 APPLN BID 1 WEEK": 0.5,
    "1 TAB BID 3 WEEK": 0.5,

    # ---------- 固定 TID ----------
    "2 TAB TID 7 DAY": 1 / 3,
    "1 CAP TID 5 DAY": 1 / 3,
}

F17 = {
    # ---- once daily ----
    "1 TAB QPM 4 WEEK": 1,
    "1 TAB QD 6 WEEK": 1,
    "1 TAB ONCE 1 MO30": 1,
    "1.5 TAB QD 100 DAY": 1,

    # ---- twice daily ----
    "1 APPLN BID 3 MO30": 0.5,
    "TWO INHALATIONS B.I.D.": 0.5,
    "TAKE ONE BID": 0.5,
    "1.5 TAB BID 3 MO30": 0.5,

    # ---- three times daily ----
    "1 PO TID": 1 / 3,
    "1 TAB TID 28 DAY": 1 / 3,
}

F18 = {
    # ---- once daily ----
    "1 UNIT QD 1 DAY": 1,
    "1 TO 2 SPRAY QD 30 DAY": 1,
    "BY MOUTH ONCE DAILY": 1,
    "1 TO 1 TAB QD 3 MO30": 1,
    "2 TAB QHS 60 DAY": 1,
    "1 TAB DAILY": 1,
    "1 TAB QD 84 DAY": 1,
    "3 TAB QHS 30 DAY": 1,
    "1 ML QD 1 DAY": 1,
    "1 TAB ONCE 1 DAY": 1,
    "1 CAP QHS 3 MO30": 1,
    "3 TAB QD 3 MO30": 1,
    "2 PUFF QD 1 MO30": 1,
    "1 TAB QD 2 MO28": 1,
    "1 CAP ONCE 1 DAY": 1,
    "1 PATCH QD 1 MO30": 1,
    "1 TAB QHS 2 WEEK": 1,

    # ---- twice daily ----
    "2 SPRAY BID 1 MO30": 0.5,
    "4 DROPS BID 7 DAY": 0.5,
    "1.5 TAB BID 1 MO30": 0.5,
    "2 SPRAY BID 30 DAY": 0.5,
    "ONE INHALATION TWICE DAILY": 0.5,

    # ---- three times daily ----
    "2 TAB TID 30 DAY": 1 / 3,

    # ---- every 4 hours (fixed, NOT PRN) ----
    "1 TO 2 PUFF Q4H 1 MO30": 1 / 6,  # 4 hours = 4/24 day
}

F19 = {
    # ---- once daily ----
    "1 TAB QAM 90 DAY": 1,
    "2 CAP QD 90 DAY": 1,
    "1 CAP QD 1 MO28": 1,
    "2 CAP QD 3 MO30": 1,
    "0.5 TO 1 TAB QD 30 DAY": 1,
    "TAKE ONE TAB DAILY": 1,
    "1.5 TAB QD 90 DAY": 1,
    "1 CAP QHS 60 DAY": 1,
    "0.5 TAB QHS 3 MO30": 1,
    "1 TAB QD 6 DAY": 1,
    "1 TO 1 TAB QD 90 DAY": 1,

    # ---- twice daily ----
    "1.5 TAB BID 30 DAY": 0.5,
    "1.5 TAB BID 100 DAY": 0.5,
    "ONE CAPSULE TWICE DAILY": 0.5,

    # ---- four times daily ----
    "5 ML QID 10 DAY": 0.25,

    # ---- weekly ----
    "ONE TABLET WEEKLY ON AN EMPTY STOMACH": 7,
    "1 FOIS/SEM": 7,  # French: once per week
}

F20 = {
    # ---- once daily (QD / QHS / QAM / wording) ----
    "2 CAP QD 60 DAY": 1,
    "1 TAB QD 12 WEEK": 1,
    "1 APPLN QHS 1 MO30": 1,
    "2 TAB QD 3 DAY": 1,
    "1 CAP QD 7 DAY": 1,
    "1 TAB QHS 15 DAY": 1,
    "1 CAP QD 14 DAY": 1,
    "1 TAB QHS 14 DAY": 1,
    "1 CAP QD 2 WEEK": 1,
    "1 TAB QD 20 DAY": 1,
    "1 TAB QD 120 DAY": 1,
    "1 TAB QD 180 DAY": 1,
    "1 TAB QHS 10 DAY": 1,
    "1 TAB QD 1 DAY": 1,
    "TAKE 1 DAILY": 1,
    "3 TAB QHS 1 MO30": 1,
    "TWO SPRAYS TO EACH NOSTRIL ONCE DAILY IN THE MORNING": 1,

    # ---- twice daily (BID) ----
    "1 TAB BID 20 DAY": 0.5,
    "2 INHLN BID 30 DAY": 0.5,
    "ONE TABLET P.O. B.I.D.": 0.5,
    "2 INHLN BID 1 MO30": 0.5,
    "1 TO 2 SPRAY BID 30 DAY": 0.5,
    "1 APPLN BID 3 WEEK": 0.5,

    # ---- three times daily (TID) ----
    "10 ML TID 10 DAY": 1 / 3,
    "2 TAB TID 1 MO30": 1 / 3,
    "2 TAB TID 10 DAY": 1 / 3,

    # ---- weekly ----
    "2X/WEEK": 3.5,  # every ~3.5 days
}

F21 = {
    # ---- weekly ----
    "ONE TABLET WEEKLY": 7,

    # ---- once daily / bedtime / every 24h ----
    "1 TO 2 TAB QHS 60 DAY": 1,
    "2 TAB QHS 100 DAY": 1,
    "ONE QHS": 1,
    "2 APPLN QD 30 DAY": 1,
    "2 TAB QD 2 MO30": 1,
    "1 APPLN ONCE 1 DAY": 1,
    "Q24H": 1,
    "1 TAB DAILY PO": 1,
    "0.5 TO 1 TAB QD 100 DAY": 1,
    "2 TAB QHS 90 DAY": 1,
    "2 SPRAY QD 21 DAY": 1,

    # ---- twice daily ----
    "1 APPLN BID 15 DAY": 0.5,
    "5 ML BID 7 DAY": 0.5,
    "2 CAP BID 30 DAY": 0.5,

    # ---- three times daily ----
    "1 CAP TID 3 MO30": 1 / 3,
    "1 TAB TID 15 DAY": 1 / 3,
    "1 TAB TID 1 WEEK": 1 / 3,
    "1 TAB TID 2 MO30": 1 / 3,

    # ---- four times daily ----
    "1 TO 2 TSP QID 10 DAY": 0.25,
    "1 TAB QID 1 MO30": 0.25, 
}

F22 = {
    # ---- once daily ----
    "0.5 TAB QHS 60 DAY": 1,
    "0.5 TAB QHS 100 DAY": 1,
    "1 CAP QHS 100 DAY": 1,
    "2 TAB QAM 1 MO30": 1,
    "2 TAB QAM 30 DAY": 1,
    "2 TAB QAM 30 DAY": 1,
    "0.65 ML ONCE 1 DAY": 1,

    # ---- every 2 days ----
    "1 TAB Q2D 30 DAY": 2,
    "1 TAB Q2D 100 DAY": 2,

    # ---- twice daily ----
    "1 SPRAY BID 1 MO30": 0.5,
    "1 CAP BID 14 DAY": 0.5,
    "1 DROPS BID 7 DAY": 0.5,

    # ---- three times daily ----
    "1 APPLN TID 7 DAY": 1 / 3,
    "4 ML TID 10 DAY": 1 / 3,
    "2 CAP TID 10 DAY": 1 / 3,

    # ---- once daily (wording) ----
    "1 BY MOUTH OD": 1,
}
F23 = {
    # ---- taper simplified to daily ----
    "TAKE TWO TABLETS ON DAY ONE AND ONE TABLET DAILY THEREAFTER UNTIL FINISHED": 1,

    # ---- once daily ----
    "2 TAB QD 28 DAY": 1,
    "0.5 ML ONCE 1 DAY": 1,
    "1 TAB QD 3 WEEK": 1,
    "1 CAP QD 4 WEEK": 1,
    "2 SPRAY QAM 30 DAY": 1,
    "1 TO 2 CAP QD 30 DAY": 1,

    # ---- twice daily ----
    "1 TO 2 TAB BID 30 DAY": 0.5,

    # ---- three times daily ----
    "2 DROPS TID 7 DAY": 1 / 3,
    "1 TAB TID 1 MO28": 1 / 3,

    # ---- four times daily ----
    "1 CAP QID 1 WEEK": 0.25,

    # ---- STAT ----
    "1 CAP STAT 1 DAY": 1,
}

F23 = {
    # ---- taper simplified to daily ----
    "TAKE TWO TABLETS ON DAY ONE AND ONE TABLET DAILY THEREAFTER UNTIL FINISHED": 1,

    # ---- once daily ----
    "2 TAB QD 28 DAY": 1,
    "0.5 ML ONCE 1 DAY": 1,
    "1 TAB QD 3 WEEK": 1,
    "1 CAP QD 4 WEEK": 1,
    "2 SPRAY QAM 30 DAY": 1,
    "1 TO 2 CAP QD 30 DAY": 1,

    # ---- twice daily ----
    "1 TO 2 TAB BID 30 DAY": 0.5,

    # ---- three times daily ----
    "2 DROPS TID 7 DAY": 1 / 3,
    "1 TAB TID 1 MO28": 1 / 3,

    # ---- four times daily ----
    "1 CAP QID 1 WEEK": 0.25,

    # ---- STAT ----
    "1 CAP STAT 1 DAY": 1,
}

F24 = {
    # ---- once daily ----
    "2 PUFF QD 30 DAY": 1,
    "1 TO 2 TAB QD 100 DAY": 1,
    "1 CAP ONCE 3 MO30": 1,
    "ONE SPRAY TO EACH NOSTRIL DAILY": 1,
    "0.5 TAB QAM 30 DAY": 1,
    "1 PATCH QD 60 DAY": 1,
    "1 CAP QD 6 MO30": 1,

    # ---- twice daily ----
    "2 CAP BID 10 DAY": 0.5,
    "2 PUFF BID 3 MO30": 0.5,
    "1 TAB BID 6 MO30": 0.5,
    "2 TAB BID 28 DAY": 0.5,
    "1 PO BID": 0.5,
    "4 SPRAY BID 1 MO30": 0.5,
    "1 PUFF BID": 0.5,
    "1 CAP BID 28 DAY": 0.5,
    "2 BID": 0.5,
    "1 TAB BID 1 YEAR": 0.5,

    # ---- three times daily ----
    "1 APPLN TID 1 MO30": 1 / 3,
    "1 TAB QID 60 DAY": 0.25,      # QID = 4/day
    "1 APPLN TID 30 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 5 DAY": 0.25,
    "1 TAB QID 3 MO30": 0.25,

    # ---- every 2 days ----
    "1 TAB Q2D 3 MO30": 2,

    # ---- weekly ----
    "3X/WEEK": 7 / 3,  # every ~2.33 days

    # ---- fixed hour-based (non-PRN) ----
    "1 PUFF Q4H 1 MO30": 1 / 6,  # every 4 hours
}

F25 = {
    # ---- once daily ----
    "2 TAB QPM 1 MO30": 1,
    "1 TAB PO OD": 1,
    "2 CAP QHS 1 MO30": 1,
    "1 TAB QAM 28 DAY": 1,
    "ONE QAM": 1,
    "1 APPLN QHS 10 DAY": 1,
    "1 TAB QD 4 DAY": 1,
    "4 TAB QD 1 MO30": 1,
    "TWO SPRAY TO EACH NOSTRIL DAILY": 1,
    "1 TO 2 TAB QD 60 DAY": 1,
    "1 TO 2 TAB QD 3 MO30": 1,
    "0.5 TAB QHS 90 DAY": 1,
    "0.5 TAB QHS 2 MO30": 1,
    "ONE CAPSULE ONCE DAILY": 1,

    # ---- twice daily ----
    "1 CAP BID 1 WEEK": 0.5,
    "1 TAB BID 21 DAY": 0.5,
    "ONE INHALATION B.I.D.": 0.5,
    "1 PUFF BID 60 DAY": 0.5,
    "PO BID": 0.5,
    "ONE TABLET BID": 0.5,
    "2 TAB BID 7 DAY": 0.5,
    "4 DROPS BID 1 WEEK": 0.5,
    "1 APPLN BID 20 DAY": 0.5,

    # ---- three times daily ----
    "1 TO 2 TAB TID 30 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 14 DAY": 0.25,
    "1 CAP QID 5 DAY": 0.25,
    "1 APPLN QID 7 DAY": 0.25,
    "2 PUFF QID 1 MO30": 0.25,

    # ---- every other day ----
    "1 TAB Q2D 30 DAY": 2,  # if present elsewhere, pattern kept consistent
}

F26 = {
    # ---- once daily ----
    "1 TO 3 TAB QHS 30 DAY": 1,
    "0.5 TO 1 TAB QD 1 MO30": 1,
    "1 APPLN QD 10 DAY": 1,
    "4 TAB QD 30 DAY": 1,
    "2 CAP QHS 30 DAY": 1,
    "1 APPLN QD 7 DAY": 1,
    "1 TAB QD 6 MO28": 1,
    "1 TAB QAM 2 WEEK": 1,
    "1 APPLN QHS 1 WEEK": 1,

    # ---- twice daily ----
    "0.5 TAB BID 2 MO30": 0.5,
    "1 APPLN BID 60 DAY": 0.5,
    "2 TABLET B.I.D. WITH MEALS": 0.5,
    "1 APPLN BID 2 MO30": 0.5,
    "2 TAB BID 2 MO30": 0.5,

    # ---- three times daily ----
    "0.5 TAB TID 1 MO30": 1 / 3,
    "2 CAP TID 7 DAY": 1 / 3,
    "1 TO 2 TAB TID 1 MO30": 1 / 3,

    # ---- four times daily ----
    "2 TAB QID 1 MO30": 0.25,
    "5 ML QID 7 DAY": 0.25,
    "2 TAB QID 1 MO30": 0.25,  
    "1 TAB QID 100 DAY": 0.25,
}

F27 = {
    # ---- once daily ----
    "1 CAP QAM 3 MO30": 1,
    "1 TAB QPM 2 WEEK": 1,
    "1 APPLN QHS 3 MO30": 1,
    "0.5 TAB QD 28 DAY": 1,
    "0.5 TAB QAM 3 MO30": 1,
    "1 CAP QD 1 WEEK": 1,
    "1 TAB OTH 1 MO30": 1,   # OTH without PRN, treated as fixed daily
    "1 TAB OTH 30 DAY": 1,   

    # ---- twice daily ----
    "2 CAP BID 1 MO30": 0.5,
    "1 TAB PO BID": 0.5,
    "1 CAP BID 2 MO30": 0.5,
    "1/2 BID": 0.5,

    # ---- three times daily ----
    "4 ML TID 7 DAY": 1 / 3,
    "7.5 ML TID 10 DAY": 1 / 3,
    "2 CAP TID 1 MO30": 1 / 3,
    "1 TAB TID 3 DAY": 1 / 3,

    # ---- four times daily ----
    "1 TAB QID 1 WEEK": 0.25,

    # ---- STAT / once ----
    "2 TAB STAT 1 DAY": 1,
}
F28 = {
    # ---- once daily ----
    "1 TAB QHS 20 DAY": 1,
    "1 APPLN QD 60 DAY": 1,
    "1 TAB QD 42 DAY": 1,
    "1 TO 2 TAB QD 90 DAY": 1,
    "1 CAP QAM 4 WEEK": 1,
    "0.5 TO 1 TAB QD 60 DAY": 1,
    "ONE TAB DAILY PO": 1,
    "1 TAB ONCE 90 DAY": 1,
    "1 PO QHS": 1,
    "1 TO 2 TAB QHS 100 DAY": 1,
    "1 TO 2 TAB QHS 3 MO30": 1,
    "1 TAB TID 20 DAY": 1 / 3,  # (moved to TID section below if you prefer)
    "1 TAB TID 4 WEEK": 1 / 3,  # (same)

    # ---- twice daily ----
    "ONE TABLET PO BID": 0.5,
    "2 SPRAY BID 2 WEEK": 0.5,
    "2 TO 4 SPRAY BID 30 DAY": 0.5,
    "1 INHLN BID 100 DAY": 0.5,
    "1 TAB BID 50 DAY": 0.5,
    "1 TAB BID 6 WEEK": 0.5,
    "3 TAB BID 1 MO30": 0.5,
    "1 TO 2 PUFF BID 1 MO30": 0.5,
    "ONE TABLET B.I.D. WITH MEALS": 0.5,
    "2 DAILY": 0.5,

    # ---- three times daily ----
    "2 DROPS TID 10 DAY": 1 / 3,
    "1 TSP TID 10 DAY": 1 / 3,
    "1 TAB TID 20 DAY": 1 / 3,
    "1 TAB TID 4 WEEK": 1 / 3,
    "6 ML TID 10 DAY": 1 / 3,
    "1 APPLN TID 1 WEEK": 1 / 3,
    "1 APPLN TID 5 DAY": 1 / 3,
    "1 APPLN TID 2 WEEK": 1 / 3,

    # ---- every 3 days ----
    "Q3D": 3,

    # ---- weekly ----
    "1X/WEEK": 7,
    "1 TAB 1X/WEEK 100 DAY": 7,

    # ---- fixed q4h (non-PRN) ----
    "ONE TABLET EVERY 4 HOURS FOR RELIEF OF PAIN": 1 / 6,
}

F29 = {
    # ---- once daily ----
    "TAKE ONE DAILY": 1,
    "1 UNIT QD 30 DAY": 1,
    "1 TO 2 CAP QHS 30 DAY": 1,
    "1 CAP QHS 2 MO30": 1,
    "3 TAB QD 100 DAY": 1,
    "3 TAB QD 60 DAY": 1,
    "0.5 TO 1 TAB QD 90 DAY": 1,
    "1 PATCH QD 90 DAY": 1,
    "2 TAB QHS 28 DAY": 1,

    # ---- twice daily ----
    "1 PUFF BID 90 DAY": 0.5,
    "1 INHLN BID 60 DAY": 0.5,
    "2 TAB BID 1 DAY": 0.5,

    # ---- every 2 days ----
    "1 TAB Q2D 90 DAY": 2,

    # ---- weekly ----
    "1 TAB 1X/WEEK 8 WEEK": 7,

    # ---- four times daily ----
    "2 DROPS QID 10 DAY": 0.25,
    "1 ML QID 12 DAY": 0.25,

    # ---- STAT ----
    "1 UNIT STAT 1 DAY": 1,

    # ---- 21/28 cycle pill (treat as daily) ----
    "1 TAB 21/28D 2 MO28": 1,
}

F30 = {
    # ---- hourly based ----
    "Q5H": 5/24,                

    # ---- once daily ----
    "2 TAB QHS 2 MO30": 1,
    "1 TAB QPM 7 DAY": 1,
    "1 TAB QD 50 DAY": 1,
    "2 SPRAY QD 14 DAY": 1,
    "1 CAP QD 15 DAY": 1,

    # ---- twice daily ----
    "TWO TABLET B.I.D. WITH MEALS": 0.5,
    "2 DROPS QID 7 DAY": 0.25,    

    # ---- three times daily ----
    "SPARINGLY TID": 1/3,
    "0.5 TAB TID 30 DAY": 1/3,

    # ---- four times daily ----
    "1 APPLN QID 5 DAY": 0.25,
    "1 APPLN QID 10 DAY": 0.25,

    # ---- STAT ----
    "1 UNIT STAT 3 DAY": 1,

    # ---- fixed duration, clear frequency ----
    "1 CAP TID 2 WEEK": 1/3,
}

F31 = {
    # ---- once daily ----
    "0.5 TO 1 TAB QHS 3 MO30": 1,
    "1 TO 2 TAB QHS 90 DAY": 1,
    "TAKE ONE TABLET QHS": 1,
    "ONE TABLET AT BEDTIME": 1,
    "1 APPLN QD 1 DAY": 1,
    "1.5 TAB QD 2 MO30": 1,
    "3 CAP QD 30 DAY": 1,
    "3 TAB QHS 3 MO30": 1,
    "0.5 TAB QPM 2 MO30": 1,
    "QNOON": 1,
    "4 SPRAY QHS 1 MO30": 1,

    # ---- twice daily ----
    "1 APPLN BID 4 WEEK": 0.5,
    "1 SPRAY BID 30 DAY": 0.5,
    "1 UNIT BID 30 DAY": 0.5,
    "1 TO 2 TAB BID 1 MO30": 0.5,
    "3 TAB BID 30 DAY": 0.5,
    "4 TAB BID 1 DAY": 0.5,
    "1.5 TAB BID 90 DAY": 0.5,
    "1 APPLN BID 5 DAY": 0.5,

    # ---- three times daily ----
    "1 APPLN TID 14 DAY": 1/3,
    "1 CAP TID 90 DAY": 1/3,
    "5  ML  PO TID": 1/3,

    # ---- four times daily ----
    "1 TAB QID 90 DAY": 0.25,

    # ---- every 2 days ----
    "1 TAB Q2D 60 DAY": 2,
    "1 TAB Q2D 1 MO30": 2,

    # ---- weekly ----
    "1 TAB 1X/WEEK 52 WEEK": 7,

    # ---- alternate day (treat as every other day) ----
    "ALT.DAY": 2,

    # ---- fixed q4-6h (non-PRN) ----
    "1 TAB Q4-6H 5 DAY": 5/24,  
}

F32 = {
    # ---- once daily ----
    "1 PATCH QD 3 MO30": 1,
    "1/2 DAILY": 1,
    "1 CAP QD 2 YEAR": 1,
    "1 CAP QD 10 DAY": 1,
    "1 TAB QD 56 DAY": 1,
    "1 SPRAY QD 30 DAY": 1,
    "1 CAP QD 4 MO30": 1,
    "1 TABLET DAILY": 1,
    "1 TABLET DAILY": 1,
    "1 TAB ONCE 30 DAY": 1,
    "1 TAB QD 5 MO30": 1,
    "1 TAB QD 56 DAY": 1,
    "1 CAP QD 10 DAY": 1,

    # ---- twice daily ----
    "2 CAP BID 7 DAY": 0.5,
    "1 CAP BID 1 MO28": 0.5,
    "1 CAP BID 2 WEEK": 0.5,
    "0.5 TAB BID 28 DAY": 0.5,
    "2 DROPS BID 7 DAY": 0.5,

    # ---- three times daily ----
    "2 CAP TID 30 DAY": 1/3,
    "2 TAB TID 1 WEEK": 1/3,

    # ---- every 3 days ----
    "1 PATCH Q3D 30 DAY": 3,

    # ---- twice per week ----
    "1 TAB 2X/WEEK 1 MO30": 3.5,

    # ---- fixed q4-6h (non-PRN) ----
    "1 TO 2 TAB Q4-6H 20 DAY": 5/24,  
}

F33 = {
    # ---- once daily ----
    "Q AM": 1,                    
    "3 CAP QD 1 MO30": 1,
    "0.5 TO 1 TAB QHS 60 DAY": 1,  
    "0.5 TO 1 TAB QD 3 MO30": 1,
    "0.5 TAB QD 6 MO30": 1,

    # ---- twice daily ----
    "1 APPLN BID 25 DAY": 0.5,
    "2 PUFF BID 90 DAY": 0.5,

    # ---- four times daily ----
    "1 ML QID 10 DAY": 0.25,

    # ---- STAT (treat as once on that day) ----
    "1 ML STAT 1 MO30": 1,

    # ---- every other day ----
    "1 APPLN OTH 1 MO30": 2,
}

F34 = {
    # ---- once daily ----
    "0.5 TAB QAM 2 MO30": 1,
    "1 CAN QD 30 DAY": 1,
    "1 TAB QD 8 WEEK": 1,
    "3 TAB QHS 60 DAY": 1,
    "1 TAB QD 2 DAY": 1,
    "TAKE ONE TABLET OD": 1,
    "1/2 OD": 1,
    "4 TAB QD 60 DAY": 1,
    "0.25 TAB QD 90 DAY": 1,

    # ---- twice daily ----
    "1 TO 2 PUFF BID 3 MO30": 0.5,
    "1 PUFF Q12H 30 DAY": 0.5,     
    "1 TAB Q12H 30 DAY": 0.5,     

    # ---- three times daily ----
    "TAKE ONE CAPSULE TID": 1/3,
    "5 ML TID 5 DAY": 1/3,
    "1 CAP TID 15 DAY": 1/3,
    "5 ML TID 1 WEEK": 1/3,
    "1 DROPS TID 7 DAY": 1/3,
    "10 ML TID 7 DAY": 1/3,
    "1 DROPS TID 1 MO30": 1/3,
    "1 CAP TID 60 DAY": 1/3,
    "2 TAB TID 3 MO30": 1/3,

    # ---- four times daily ----
    "1 TO 2 INHLN QID 2 MO30": 0.25,
    "ONE QID": 0.25,
    "1 TO 2 PUFF QID 30 DAY": 0.25,
    "10 ML QID 2 WEEK": 0.25,
    "1 TO 2 DROPS QID 10 DAY": 0.25,

    # ---- every other day ----
    "1 TAB OTH 60 DAY": 2,

    # ---- 2x per week ----
    "1 TAB 2X/WEEK 3 MO30": 3.5,

    # ---- weekly ----
    "1 UNIT 1X/WEEK 2 WEEK": 7,

    # ---- every 6 hours (fixed, non-PRN) ----
    "1 TAB Q6H 30 DAY": 0.25,      
    "1 TAB Q6H 30 DAY": 0.25,

    # ---- STAT / ONCE ----
    "1 TAB STAT 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "1 TAB ONCE 1 DAY": 1,         
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
    "2 TAB ONCE 1 DAY": 1,
}

F35 = {
    # ---- fixed frequencies (no PRN / no VARIABLE / no 0) ----
    "1 INHLN QD 30 DAY": 1,
    "1.5 TAB BID 60 DAY": 0.5,
    "2 TAB QD 2 YEAR": 1,
    "1 TAB 21/28D 28 DAY": 1,  
    "1 PUFF QD 3 MO30": 1,
    "2.5 TAB QD 60 DAY": 1,
    "1 TAB BID 4 MO30": 0.5,
    "1 CAP BID 15 DAY": 0.5,
    "1 DROPS BID 30 DAY": 0.5,
    "1 APPLN BID 1 YEAR": 0.5,
    "2 PO OD": 1,
    "0.5 TO 1 TAB OTH 1 MO30": 2,
    "1 APPLN BID 6 WEEK": 0.5,
    "1 TAB BID 12 MO30": 0.5,
    "2 TAB QHS 4 WEEK": 1,
    "2 TAB TID 5 DAY": 1/3,
    "4 TAB QHS 1 MO30": 1,
    "1 TAB QD 40 DAY": 1,
    "1 CAP QHS 28 DAY": 1,
}

F36 = {
    # weekly / monthly
    "1 TAB 1X/WEEK 1 YEAR": 7,
    "1 CAP TID 1 WEEK": 1/3,
    "1 TAB QAM 1 WEEK": 1,
    "1X/MONTH": 30,

    # daily / multiple daily
    "2 CAP QAM 1 MO30": 1,
    "2 TAB TID 90 DAY": 1/3,
    "2 TAB QD 1 MO28": 1,
    "1 TAB BID PO": 0.5,
    "1 TAB BID PO ": 0.5,  
    "1 TAB BID PO": 0.5,
    "1 TAB Q4-6H 10 DAY": 4,        
    "1 TAB Q4-6H 10 DAY ": 4,
    "1 TO 2 TAB TID 10 DAY": 1/3,
    "1 DROPS QID 7 DAY": 0.25,
    "1 TO 1 TAB QPM 30 DAY": 1,
    "1 TO 2 TAB QID 10 DAY": 0.25,
    "2 DROPS QID 1 WEEK": 0.25,
    "2 DROPS QID 5 DAY": 0.25,
    "1 TAB Q12H 60 DAY": 0.5,
    "1 UNIT QD 3 DAY": 1,
    "1 TAB 3X/WEEK 100 DAY": 7/3,    
    "1 CAP QID 30 DAY": 0.25,
    "1 CAP ONCE 1 MO30": 1,
    "0.5 TAB BID 1 MO28": 0.5,
    "0.5 TAB QD 1 MO28": 1,
    "1 CAP QAM 2 MO30": 1,
    "1 ML OTH 1 DAY": 2,
    "1 TO 3 TAB QHS 3 MO30": 1,      
    "1.5 TAB QHS 3 MO30": 1,
    "1 TO 2 TAB QD 2 MO30": 1,
    "1 CAP ONCE 1 MO30 ": 1,
    "TWO CAPSULES TWICE DAILY": 0.5,
}

F37 = {
    # --- weekly / monthly ---
    "1 TAB 1X/WEEK 12 MO30": 7,
    "ONE TABLET ONCE WEEKLY ON EMPTY STOMACH": 7,
    "1 ML 1X/MONTH 12 MO30": 30,

    # --- fixed schedules (no PRN, no 0) ---
    "1 TO 2 CAP QD 1 MO30": 1,
    "1.5 TAB QAM 1 MO30": 1,
    "3 TAB QD 90 DAY": 1,
    "1 TAB QD 45 DAY": 1,
    "1 SUP BID 5 DAY": 0.5,
    "1 TAB ONCE 5 DAY": 1,
    "10 ML BID 10 DAY": 0.5,
    "3 TAB QHS 100 DAY": 1,
    "2 TAB QD 5 DAY": 1,
    "2 TAB BID 5 DAY": 0.5,
    "2 TAB QD 5 DAY ": 1,
    "1 CAP QD 2 DAY": 1,
    "1 CAP QD 3 DAY": 1,
    "1 TAB QID 2 WEEK": 0.25,  # QID = 4 times/day => interval 0.25 day

    # --- hour-based but fixed (no PRN) ---
    # none here that are non-PRN; your hour-based ones all contain PRN
}

freq_map_list = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
                  F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26
                  , F27, F28, F29, F30, F31, F32, F33, F34, F35, F36, F37]

def map_frequency_to_numeric(df, freq_map_list = freq_map_list):
    if "Event_interval_days" not in df.columns:
        df["Event_interval_days"] = pd.NA

    freq_series = (
        df["Frequency"]
        .astype("string")
        .str.upper()
    )
    for freq_map in freq_map_list:
        mapped = (freq_series.map(freq_map)
        .astype("Float64"))
        df.loc[df["Event_interval_days"].isna(), "Event_interval_days"] = mapped

    return df
