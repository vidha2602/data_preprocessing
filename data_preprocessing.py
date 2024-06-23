import openai
import json
import ast
import os
import chainlit as cl
import requests
import os
import time
import io
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import langchain
import openai
import os
from langchain.tools.python.tool import PythonAstREPLTool
import time
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from googleapiclient.discovery import build
from google.oauth2 import service_account

import pandas as pd
openai.api_key = "YOUR_API_KEY"
os.environ['OPENAI_API_KEY']=openai.api_key
engine = "text-embedding-ada-002"  # Specify the OpenAI engine


# Set the environment variable
os.environ['CHAINLIT_AUTH_SECRET'] = 'YOUR_CHAINLIT_AUTH_SECRET'

from typing import Optional
import chainlit as cl

MAX_ITER = 5
python_repl = PythonAstREPLTool()
def repl_tool_fun(code):
    output= python_repl.run(code)
    print("output: output: ", output)
    return str(output)

functions = [
  {'name': 'run_python_code',
 'description': f'''Useful if you need to run python code for doing data analysis and plotting graphs.''',
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Python code to run to do required analysis.",
                },
            },
            "required": ["query"]
        },
    },
    {'name': 'run_embedding_search',
     'description': 'Useful if you want to search the question on text embeddings to get context and then answer.',
     "parameters": {"type":"object",
                    "properties": {
                                 "query":{
                                        "type":"string",
                                        "description": "Exact User question that need to be searched in the vector space to get relevant context and then answer. Note that this query should be an exact question of the user."
                                        }
                                }
                    }
     }
]

async def process_new_delta(new_delta, openai_message, content_ui_message, function_ui_message):
    if "role" in new_delta:
        openai_message["role"] = new_delta["role"]
    if "content" in new_delta:
        new_content = new_delta.get("content") or ""
        openai_message["content"] += new_content
        await content_ui_message.stream_token(new_content)
    if "function_call" in new_delta:
        if "name" in new_delta["function_call"]:
            openai_message["function_call"] = {
                "name": new_delta["function_call"]["name"]}
            await content_ui_message.send()
            function_ui_message = cl.Message(
                author=new_delta["function_call"]["name"],
                content="", indent=1, language="json")
            await function_ui_message.stream_token(new_delta["function_call"]["name"])

        if "arguments" in new_delta["function_call"]:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
            await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
    return openai_message, content_ui_message, function_ui_message

@cl.on_chat_start
async def start_chat():
    files = []

    # Wait for the user to upload two Excel files
    while len(files) != 2:
        if len(files)==1:
            confirm_msg = cl.Message(content="First file uploaded successfully.")
            await confirm_msg.send()
            file_response = await cl.AskFileMessage(
                content="Please Upload second file.",
                accept=["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
                max_size_mb=50,
                timeout=180,
                ).send() 


        else:
            file_response = await cl.AskFileMessage(
                content="Please upload two Excel files to begin!",
                accept=["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"],
                max_size_mb=50,
                timeout=180,
            ).send()

        
        # Extract the first file from the response and append to files list
        files.append(file_response[0])

    processing_msg = cl.Message(content="Files are being processed, please wait...")
    await processing_msg.send()
    await cl.sleep(1)




    file_1, file_2 = files[0], files[1]
    xl_file_1 = io.BytesIO(file_1.content)
    xl_file_2 = io.BytesIO(file_2.content)
    df1 = pd.read_excel(xl_file_1)
    df2 = pd.read_excel(xl_file_2)

    cl.user_session.set("df1", df1)
    cl.user_session.set("df2", df2)

    df1.to_excel(file_1.path)
    df2.to_excel(file_2.path)

    loading_code = f"""\n
import pandas as pd 
xl_file_1 = '{file_1.path}'
xl_file_2 = '{file_2.path}'
df1=pd.read_excel(xl_file_1)
df2=pd.read_excel(xl_file_2)
print('loading code over...')

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import openai
import time
engine = "text-embedding-ada-002"
openai.api_key = "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"
# Function to get text embeddings


def merge_dataframes(df1, df2, key_column):
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    import openai
    import time
    import pandas as pd

    def get_embedding(lst):
        import openai
        import time
        import pandas as pd
        engine = "text-embedding-ada-002"
        openai.api_key = "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"
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

        final_embeddings = []
        for k in range(0, len(lst), 100):
            texts_lst = lst[k:k+100]
            embeddings = []
            for i in range(len(texts_lst)):
                if len(texts_lst[i]) <= 1:
                    texts_lst[i] = "   "
            try:
                dic_lst = openai.Engine(id=engine).embeddings(input=texts_lst)['data']
            except:
                print("Rate limit error occurred requesting again after 10 sec")
                time.sleep(60)
                dic_lst = openai.Engine(id=engine).embeddings(input=texts_lst)['data']
            for i in range(len(texts_lst)):
                embeddings.append(dic_lst[i]["embedding"])
            final_embeddings.extend(embeddings)
        return final_embeddings


    values_df_1_lst = df1[key_column].tolist()
    values_df_2_lst = df2[key_column].tolist()

    universal_lst = values_df_1_lst + values_df_2_lst
    universal_lst_unique = list(set(universal_lst))

    embedding_dic = get_embedding(universal_lst_unique)

    similarity_scores = []
    for value1 in values_df_1_lst:
        for value2 in values_df_2_lst:
            index1 = universal_lst_unique.index(value1)
            index2 = universal_lst_unique.index(value2)
            embedding1 = embedding_dic[index1]
            embedding2 = embedding_dic[index2]
            similarity_score = sklearn_cosine_similarity([embedding1], [embedding2])[0][0]
            similarity_scores.append([value1, value2, similarity_score])

    scores_df = pd.DataFrame(similarity_scores, columns=[key_column + "_1", key_column + "_2", "Similarity"])
    idx = scores_df.groupby(key_column + "_1")['Similarity'].idxmax()
    highest_similarity_df = scores_df.loc[idx].reset_index(drop=True)
    merged_df1 = pd.merge(highest_similarity_df, df1, how='left', left_on=key_column + '_1', right_on=key_column)
    merged_final = pd.merge(merged_df1, df2, how='left', left_on=key_column + '_2', right_on=key_column, suffixes=('', '_drop'))

    to_drop = [col for col in merged_final if col.endswith('_drop') or col == key_column + '_2']
    merged_final.drop(to_drop, axis=1, inplace=True)
    if key_column in merged_final.columns:
        merged_final = merged_final.loc[:, ~merged_final.columns.duplicated()]


    return merged_final

print('loading code over...')
    """
    print("running loading code...")
    repl_tool_fun(loading_code)


    msg = cl.Message(content=f"Two files have been uploaded successfully. \n First file - {file_1.path}\n Second file  - {file_2.path}. \n Start giving commands like - Merge, Append ")
    await msg.send()
    prompt = f"""You are data analyst/scientist who helps in Data preprocessing like merging or concatanating/appending the provided dataframes based on the command given by the user. You should use run_python_code function to execute the Python code.
    If the user commands you to Merge, follow these instructions:  
       1. A function named 'merge_dataframes' is available, you can call this function wherever required. This funtion takes the two dataframes and the key column as input arguments and returns a merged dataframe based on the similarity of embeddings for values in a specified key column. 
       2. You need to find the most appropriate key_column from the excel files and confirm from the user if the key_column you found is the necessary key_column. 
       3. You must also save the final merged dataframe as excel file with name 'result_table.xlsx', with index set to False, always. 
       4. df1 and df2 are aleady loaded with excel files, no need to read them again.
       5. Variable persistent, So you can directly use old code varibales.
       6. Only display the first 3 entries of the result_table.xlsx . 
       7. Always import the necessary libraries. 

    If the user commands you to Append or Concatenate, follow these instructions: 
       1. First step is to find semantically similar columns in two pandas dataframes df1, df2. You need to figure out and display the potential semantically similar columns in both the dataframes. Use your intelligence instead of writing python code for finding the semantically similar columns.
       2. Write python code to create new dataframe for concating semantically similar columns which we get from step-1 above. use 'pd.concat' method while appending.
       3. You must also save the final concatenate dataframe as excel file with name 'result_table.xlsx'.
       4. df1 and df2 are aleady loaded with excel files, no need to read them again.
       5. Variables are persistent in Python environment, So you can directly use old code varibales.
       6. Write python code in one String i.e use \n characters in generated python code to separate code lines but don't insert them in new lines.
       7. Column Names dataframe1(df1):{df1.columns}
          Column Names Dataframe2(df2):{df2.columns}

    """

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": prompt}]
        )



@cl.on_message
async def run_conversation(user_message: str):
    message_history = cl.user_session.get("message_history")
    print(len(message_history), 'at start:')

    print(message_history)

    for message in message_history:
        if message['role'] == 'function':
            message['content'] = ""

    # if len(message_history) > 20:
    #     del message_history[9:len(message_history)-9]

    message_history.append({"role": "user", "content": user_message})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
            model="gpt-4-turbo", #"gpt-4-0613",
            messages=message_history,
            stream=True,
            function_call="auto",
            functions=functions,
            temperature=0
        ):
            new_delta = stream_resp.choices[0]["delta"]
            openai_message, content_ui_message, function_ui_message = await process_new_delta(
                new_delta, openai_message, content_ui_message, function_ui_message)

        message_history.append(openai_message)
        if function_ui_message is not None:
            await function_ui_message.send()

        if stream_resp.choices[0]["finish_reason"] == "stop":
            break
        elif stream_resp.choices[0]["finish_reason"] != "function_call":
            raise ValueError(stream_resp.choices[0]["finish_reason"])

        function_name = openai_message.get("function_call").get("name")
        arguments = ast.literal_eval(
            openai_message.get("function_call").get("arguments"))

        if function_name == 'run_python_code':
            function_response = ""
            try:
                function_response = repl_tool_fun(arguments.get("query"))
            except Exception as e:
                function_response = f"An error occurred: {str(e)}"
                await cl.Message(content=function_response).send()

            message_history.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )

            await cl.Message(
                author=function_name,
                content=str(function_response),
                language="json",
                indent=1,
            ).send()
        cur_iter += 1

    filename = "result_table.xlsx"
    if os.path.exists(filename):
        elements = [
            cl.File(
                name="result_table.xlsx",
                path="./result_table.xlsx",
                display="inline",


            ),
        ]

        await cl.Message(
            content="Click to download the excel file:", elements=elements
        ).send()
        os.remove(filename)





