import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.chains import RunnableSequence

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def getLLM():
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    return llm

prompt_template = """
你是一个非常聪明的助手，能够回答用户指定地点指定时间内的纪念日日期和纪念日名称，并以 JSON 格式返回结果。

问题：{question}

请确保你的回答是针对指定月份和年份的纪念日，并且以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "date": "yyyy-mm-dd",
                "name": "纪念日名称"
        }},
        {{
                "date": "yyyy-mm-dd",
                "name": "纪念日名称"
        }}
        # 继续列出该月的所有节假日
    ]
}}

请确保回答的纪念日的时间和用户的一致，例如，用户问 "2024年台灣1月紀念日有哪些?" 时，你应该返回2024年台湾1月的所有紀念日，而不是10月的，或者非台灣地区的，或者非2024年的等等。
"""

def generate_hw01(question):
    llm = getLLM()
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    response = (prompt | llm).invoke({"question": question})
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    #response = chain.invoke({"question": question})
    # 6. 提取 response 中的内容并清洗
    response_content = response['text'] if isinstance(response, dict) else response.content

    # 7. 去除 JSON 格式的代码块标记部分
    json_content = response_content.strip('```json\n').strip('```')

    # 8. 将结果解析为 JSON
    try:
        parsed_result = json.loads(json_content)
        print("返回的 JSON 结果：")
        json_string = json.dumps(parsed_result, indent=4, ensure_ascii=False)
        #print(json_string)  # 美化输出，确保中文显示正常
    except json.JSONDecodeError as e:
        print("解析 JSON 时出错:", e)
    return json_string
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass


    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
