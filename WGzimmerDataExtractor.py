# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:08:17 2023

Extract Metadata from PDF of WGZimmer Emails
Using splitting on strings and OpenAI ChatGPT 3.5 ChatCompletion

@author: Cyprien Hoelzl
"""
# importing all the required modules
import PyPDF2
import openai
import pandas as pd
import yaml
import os
#%% Functions
def read_prepare_pdf(filename = "data/AntwortenWGZimmer1Guggach_Stand16-05-23_180500.pdf"):
    """
    Read PDF into a list of application texts

    Parameters
    ----------
    filename : path to file. The default is "data/AntwortenWGZimmer1Guggach_Stand16-05-23_180500.pdf".

    Returns
    -------
    stext : split text with each single application.

    """
    # creating a pdf reader object
    reader = PyPDF2.PdfReader(filename)
    
    # create one long text file
    text = ''
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text()[len(str(i)):]

    # split by person
    stext = sorted(text.split('\nFrom: no-reply@wgzimmer.ch')[1:])
    
    # remove ad
    remove_ads = "Denkst Du, dass dieses Email eine Fälschung ist oder ein Betrugversuch? Dann melde \nuns dieses Inserat! Sende diese Mail an abuse@wgzimmer.ch"
    stext = [i.split(remove_ads)[0] for i in stext]
    return stext
def extract_metadata(text0):
    """
    Split the text into metadata of the application using predefined splitting keys

    Parameters
    ----------
    text0 : input text

    Returns
    -------
    information : dictionary with informations.

    """
    keys_recursivesplit = {"none":"\nSent: ","Sent":"\nTo: ","To":"\nSubject: ","Subject":"\nName: ","Name":"\nTelefon: ","Telefon":"\nNachricht: "}
    
    text01 = text0 
    information = {}
    
    try:
        for key in keys_recursivesplit.keys():
            if key == "Name":
                information[key] = text01.split(keys_recursivesplit[key])[0].split('\n')[0]
            elif key == "Sent":
                information[key] = text01.split(keys_recursivesplit[key])[0].split('Sent: ')[1]
            elif key == 'none':
                continue
            elif key == "Telefon":
                information[key] = text01.split(keys_recursivesplit[key])[0]
                information["Nachricht"] = text01.split(keys_recursivesplit[key])[1]
            else:   
                information[key] = text01.split(keys_recursivesplit[key])[0]
            text01 = key.join(text01.split(keys_recursivesplit[key])[1:])  
    except Exception as e:
        print("Exception: {}".format(e))
    return information

def extract_data_with_OpenAI(df,filename = "data/responsesChatGPT.pickle", regenerate = False):
    """
    Extract the data with OpenAI
        information: {"Country of origin":?,
                      "Gender":?,
                      "favourite sports":?, 
                      "favourite activities":?,
                      "clean person": Yes/No?, 
                      "student": Yes/No, 
                      "age": ?, 
                      "current activities": bachelor, master, phd, working, ?, 
                      "study field": ?, 
                      "smoker":Yes/No?}
    Parameters
    ----------
    df : dataframe with column named "Nachricht".
    filename : presaved results. The defaults is "data/responsesChatGPT.pickle".
    regenerate : If True, will regenerate the GPT output. The default is False.

    Returns
    -------
    responses : dictionary with key (index) and value (extracted json string).

    """
    if regenerate == True:
        responses = {}
        for idx, row in df.iterrows():
            print("Current Applicant ID: {}".format(idx))
            message = row.Nachricht
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a dictionary generator"},
                        {"role": "user", "content": '"Message: {}"'.format(message) + '\nAnswer with a dictionary file, {"Country of origin":?,"Gender":?,"favourite sports":?, "favourite activities":?,"clean person": Yes/No?, "student": Yes/No, "age": ?, "current activities": bachelor, master, phd, working, ?, "study field": ?, "smoker":Yes/No?} Assume the gender from the name"'},
                    ]
            )
            responses[idx] = response  
        pd.to_pickle(responses,filename)
    else:
        responses = pd.read_pickle(filename)    
    return responses


def make_rating_with_OpenAI(df,filename = "data/responsesRatingsChatGPT.pickle", regenerate = False):
    """
    Rate the message with OpenAI
        information: {"Country of origin":?,
                      "Gender":?,
                      "favourite sports":?, 
                      "favourite activities":?,
                      "clean person": Yes/No?, 
                      "student": Yes/No, 
                      "age": ?, 
                      "current activities": bachelor, master, phd, working, ?, 
                      "study field": ?, 
                      "smoker":Yes/No?}
    Parameters
    ----------
    df : dataframe with column named "Nachricht".
    filename : presaved results. The defaults is "data/responsesRatingsChatGPT.pickle".
    regenerate : If True, will regenerate the GPT output. The default is False.

    Returns
    -------
    responses : dictionary with key (index) and value (extracted json string).

    """
    if regenerate == True:
        responses = {}
        for idx, row in df.iterrows():
            print("Current Applicant ID: {}".format(idx))
            message = row.Nachricht
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a rater, returning a dictionary (json) with ratings between 1-10 based on a set of criteria."},
                        {"role": "user", "content": '"Message: {}"'.format(message) + '\nAnswer with a dictionary file. Please rate the previous text based on the following criteria: {"originality":1-10,"personality":1-10,"language": 1-10}'},
                    ]
            )
            responses[idx] = response  
        pd.to_pickle(responses,filename)
    else:
        responses = pd.read_pickle(filename)    
    return responses

def put_openai_data_intodictionary(responses):
    """
    SafeLoading of json string into a dictionary object

    Parameters
    ----------
    responses : response dictionary from OpenAI.

    Returns
    -------
    data : dictionary of applicant data.

    """
    data = {}
        
    for key in responses.keys():
        try:
            response = responses[key]
            message = response["choices"][0]["message"]['content'].replace(": ?,",': "Unknown",').replace('"?" (not specified)',"Unknown")
            message_without_extra_text = message[message.find("{"):(len(message)-message[::-1].find('}'))]
            ld = yaml.load(message_without_extra_text, yaml.SafeLoader)
            ld = {key.lower().replace('favorite', 'favourite'): value for key, value in ld.items() if (('tichu' in key.lower())==False) & (('hanging out with' in key.lower())==False)}
            data[key] = ld
        except Exception as e:
            print("Exception on no {}: {}, \n\n\t{}".format(key, e,response["choices"][0]["message"]['content']))
            pass
    return data 
#%% Main code
if __name__ == '__main__':
    # Read pdf into a list of splitted texts
    demo = True
    if demo == True:
        stext = []
        for fn in ['./data/'+i for i in os.listdir('./data') if i.startswith('application_no')]:
            with open(fn, encoding = 'utf-8') as f:
                stext.append(f.read()) 
    else:
        stext = read_prepare_pdf()
    # Make dataframe with user metadata
    textmeta = [extract_metadata(text0) for text0 in stext]
    # Make dataframe
    df = pd.DataFrame(textmeta)
    # Drop duplicates
    df = df.drop_duplicates(subset = ["Name",'Nachricht'])
    # Drop if no name or phone number (not a WGZimmer text)
    df = df[((df.Name == '') & (df.Telefon == ''))==False]
    
    #%% OpenAI Chat GPT
    # https://platform.openai.com/account/usage usage for cost
    if os.path.isfile(".apikey"):
        with open(".apikey") as f:
            openai.api_key = f.readline()
    else:
        raise Warning("Missing API key file: create a file named '.apikey.txt' with the api key for openai")
    # Add origin, gender, hobbies, clean, ...
    regenerate = False
    if demo == True:
        responses = extract_data_with_OpenAI(df, filename = "data/responses_ChatGPT_demo.pickle", regenerate = regenerate)
    else:
        responses = extract_data_with_OpenAI(df, regenerate = regenerate)
    #%% Extract Json information
    data = put_openai_data_intodictionary(responses)
    # Make dataframe        
    people_data = pd.DataFrame(data).T
   
    
    #%% Merge with overall dataset
    df_expanded = df.merge(people_data, left_index = True, right_index = True)
    #%% Cleanup prompts from ChatGPT
    df_expanded['clean person'] = df_expanded['clean person'].replace(True, 'Yes').replace(False,'No')
    df_expanded['student'] = df_expanded['student'].replace(True, 'Yes').replace(False,'No')
    df_expanded['smoker'] = df_expanded['smoker'].replace(True, 'Yes').replace(False,'No')
    df_expanded['age'] = pd.to_numeric(df_expanded.age, errors='coerce')
    df_expanded['study field'] = df_expanded['study field'].str.lower().replace("not specified", None).replace("unspecified",None).replace('unknown',None)
     
    #%%Get ratings
    regenerate = False
    if demo == True:
        responses = make_rating_with_OpenAI(df, filename = "data/responsesRatings_ChatGPT_demo.pickle", regenerate = regenerate)
    else:
        responses = make_rating_with_OpenAI(df, regenerate = regenerate)
    #%% Extract Json information
    data = put_openai_data_intodictionary(responses)
    # Make dataframe        
    people_ratings = pd.DataFrame(data).T
    # Ratings merged
    df_expanded2 = df_expanded.merge(people_ratings, left_index = True, right_index = True, how = 'left')
    
    #%% Save data
    df_expanded2.to_csv('data/extracted_metadata_demo.csv')
