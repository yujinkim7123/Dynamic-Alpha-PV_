prompts = {
    "general": {
        "background_template": '''You are an expert in Psychometrics, especially {}. I am conducting the {} test on someone. I am gauging his/her position on the {} dimension through a series of open-ended questions. For clarity, here's some background this particular dimension:
===
{}
===

I am an experimenter. I've invited a participant, and we had many conversations in {}. I will input the conversation.

Please help me assess participant's score within the {} dimension of {}. 
''',
    "two_score_output": '''You should provide the percentage of each category, which sums to 100%, e.g., 30% A and 70% B. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": {{ "{}": <percentage 1>, "{}": <percentage 2> }} (The sum of percentage 1 and percentage 2 should be 100%. Output with percent sign.) 
}}''',
    "one_score_output": '''You should provide the score of participant in terms of {}, which is a number between 1 and 5. 1 denotes 'not {} at all', 3 denotes 'neutral', and 5 denotes 'strongly {}'. Other numbers in this range represent different degrees of '{}'. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": <your score>
}}'''
    },
}