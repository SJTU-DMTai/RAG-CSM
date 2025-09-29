

def get_generation_config(data_name, naive=False):
    if data_name in ['nq', 'triviaqa', 'webqa', 'hotpotqa', '2wikimultihopqa']:
        system_prompt = '''Answer the question based on the given documents.
                        Only give me the answer and do not output any other words.
                        Here is an example:
                        Question: total number of death row inmates in the us?
                        Answer: 2,718'''
        user_prompt = "Question: {question}\nAnswer:"
        max_new_tokens = 32
        metrics = ['em', 'f1']
    elif data_name == 'asqa':
        # long-form qa
        system_prompt = '''Answer the question based on the given documents.
                        The answer should be detailed.
                        Here is an example:
                        Question: When does the new bunk'd come out?
                        Answer: the new bunk'd episode 41 comes out on April 21, 2017, episode 42 comes out on April 28, 2017 and episode 42 is due to come out on May 24, 2017.
                        '''
        user_prompt = "Question: {question}\nAnswer:"
        max_new_tokens = 256
        metrics = ['f1']
    elif data_name == 'fever':
        # fact verification
        system_prompt = '''Your task is to determine whether the following claim is supported by evidence from Wikipedia based on the given documents.
                        The answer should be either 'SUPPORTS' or 'REFUTES'.
                        Only give me the answer and do not output any other words.
                        Here is an example:
                        Claim: Roman Atwood is a content creator.
                        Answer: SUPPORTS.'''
        user_prompt = "Claim: {question}\nAnswer:"
        max_new_tokens = 16
        metrics = ['acc']
    elif data_name == 'wow':
        # dialogue generation
        system_prompt = '''Your task is to generate a response to the given dialogue history based on the given documents.
                        Dialogue history:
                             "The Viking are sea pirates!\nI see! Didn't they speak the Norse language?\nWhat's the Norse language? What country speaks such?\nThe North Germans!\nSo what do the Vikings do ?are they a cult group?"
                        Answer: "They raided and trader across wide areas of Europe."'''
        user_prompt = "Dialogue context: {question}\nAnswer:"
        max_new_tokens = 64
        metrics = ['f1']
    elif data_name == 'truthful_qa':
        # muti-choice
        system_prompt = '''Choose the correct option to the question based on the documents.
                        Output only that option and do not output any other words.
                        Here is an example:
                        Question: Why do veins appear blue?
                        Options:
                            Veins appear blue due to how blue and red light penetrate human tissue\n
                            Veins appear blue because blue light has a shorter wavelength than red light\n
                            Veins appear blue because blue light does not penetrate deeply into human tissue\n
                            Veins appear blue because blue light is scattered or reflected before reaching the vein\n
                        Answer: Veins appear blue because blue light has a shorter wavelength than red light'''
        user_prompt = "Question: {question}\nAnswer:"
        max_new_tokens = 16
        metrics = ['acc']
    if not naive:
        system_prompt += "\nThe following are reference documents.\n\n{reference}"
    batch_size = 2 if data_name == 'asqa' else 8
    return system_prompt, user_prompt, max_new_tokens, metrics, batch_size
