def decision_engine(state, intensity, stress, energy, time_of_day):

    if state == "overwhelmed":
        if intensity >= 4:
            return "box_breathing", "now"
        return "journaling", "within_15_min"

    if state == "restless":
        if energy > 6:
            return "movement", "within_15_min"
        return "grounding", "now"

    if state == "calm":
        return "deep_work", "now"

    if state == "focused":
        return "deep_work", "now"

    if state == "neutral":
        return "light_planning", "later_today"

    if state == "mixed":
        return "journaling", "tonight"

    return "pause", "now"