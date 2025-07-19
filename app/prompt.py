def build_question_prompt(startup_info):
    return f"""<s>[INST] <s>[INST] Analyze this startup:\n
Name: {startup_info['name']}
Industry: {startup_info['industry']}
Pitch: {startup_info['pitch']}
Founded Year: {startup_info['year']}
Funding: {startup_info['funding']} [/INST]
Ask 10 smart questions (7 VC-style + 3 industry-specific). Include 1 question to judge founder capability. [/INST]
"""


def build_evaluation_prompt(startup_info, founder_answers):
    formatted_answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(founder_answers)])

    return f"""<s>[INST] Analyze this startup and give a full evaluation across 21 VC metrics:

Startup Info:
Name: {startup_info['name']}
Industry: {startup_info['industry']}
Pitch: {startup_info['pitch']}
Founded Year: {startup_info['year']}
Funding: {startup_info['funding']}

Founder Answers:
{formatted_answers}

Give detailed analysis with score (1–10), strengths, weaknesses, and improvement tips for each metric.
End with a 6–8 line Overall Evaluation paragraph. [/INST]\n"""
