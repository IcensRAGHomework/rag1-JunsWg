import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.chains import RunnableSequence
import subprocess

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
calendarific_api_key = "F3k6kLzjG1WjphHwkjsrsxdg0FMIs9ss"

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


prompt_hw02_extract_template = """
你是一个非常聪明的助手，能够根据用户关于纪念日，节假日等的提问提取全部指定的地点以及对应的时间，如果用户提问的时间类似于问今年，下个月这类的时间，请根据当前的时间正确换算，并以 JSON 格式返回结果。

问题：{question}

请确保你的回答正确提取全部地点的月份和年份，如果没有提取到年份，默认为2025年，如果没有指定月份请用null字符串替换，地点需要按ISO 3166-1 alpha-2转换 如果没有地点请用cn替换，并且以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "location": "地点名称的ISO 3166-1 alpha-2编码",
                "year": "yyyy"
                "month": "mm"
        }},
        {{
                "location": "地点名称的ISO 3166-1 alpha-2编码",
                "year": "yyyy"
                "month": "mm"
        }}
        # 继续列出所有地点地点对应的年份和月份
    ]
}}

请确保回答所有的地点和时间，使用ISO 3166-1 alpha-2编码转换地点，按人类的使用习惯转换时间，你应该根据今年是2025年换算,明年是2026年，去年是2024年，前年是2023年，如果没有年份或者换算不出来就用2025年替换，并确保它们的一致性，例如，用户问 "今年年台灣地区1月和美国2024年5月以及韩国明年的紀念日有哪些?" 时，返回tw，2025，01和us，2024，05,和kr，2026，null。
"""

def hw02_extract_time_location(question):
    llm = getLLM()
    prompt = PromptTemplate(input_variables=["question"], template=prompt_hw02_extract_template)
    response = (prompt | llm).invoke({"question": question})
    # 6. 提取 response 中的内容并清洗
    response_content = response['text'] if isinstance(response, dict) else response.content

    # 7. 去除 JSON 格式的代码块标记部分
    json_content = response_content.strip('```json\n').strip('```')

    # 8. 将结果解析为 JSON
    try:
        parsed_result = json.loads(json_content)
        #print("返回的时间地点的 JSON 结果：")
        #json_string = json.dumps(parsed_result, indent=4, ensure_ascii=False)
        #print(json_string)  # 美化输出，确保中文显示正常
    except json.JSONDecodeError as e:
        print("解析 JSON 时出错:", e)
        return None
    return parsed_result

prompt_hw02_show_template = """
你是一个非常聪明的助手，能够根据JSON 格式的内容提取里面的各个name和iso并以JSON格式返回结果。

问题：{question}

请确保你的回答正确提取JSON格式内容里面的list中全部，将name的内容翻译成中文后以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "date": "iso",
                "name": "name翻译成中文"
        }},
        {{
                "date": "iso",
                "name": "name翻译成中文"
        }}
        # 继续列出所有iso和翻译成中文的name
    ]
}}
只回答JSON内容即可
"""

def generate_hw02(question):
    location_time = hw02_extract_time_location(question)
    # 获取 "Result" 列表
    result = location_time['Result']
    llm = getLLM()
    if result is not None:
        # 遍历 "Result" 列表并打印每个条目
        for entry in result:
            location = entry['location']
            year = entry['year']
            month = entry['month']
            #print(f"Location: {location}, Year: {year}, Month: {month}")
            if month is None:
                url = (
                    "https://calendarific.com/api/v2/holidays?&api_key={calendarific_api_key}&country={location}&year={year}").format(
                    calendarific_api_key=calendarific_api_key, location=location, year=year)
            else:
                url = (
                    "https://calendarific.com/api/v2/holidays?&api_key={calendarific_api_key}&country={location}&year={year}&month={month}").format(
                    calendarific_api_key=calendarific_api_key, location=location, year=year,
                    month=month)
            #print(f"url: {url}")
            curl_command = ["curl", "-G", url]

            # 执行 curl 命令并获取返回的结果
            result = subprocess.run(curl_command, capture_output=True, text=True)

            # 获取输出（内容）
            if result.returncode == 0:
                #print("Response:", result.stdout)
                prompt = PromptTemplate(input_variables=["question"], template=prompt_hw02_show_template)
                response = (prompt | llm).invoke({"question": result.stdout})
                response_content = response['text'] if isinstance(response, dict) else response.content
                #print(f"返回的时间地点的 JSON 结果：{response_content}")
                json_content = response_content.strip('```json\n').strip('```')
                #print("response_content:", response_content)
                #print("strip:", json_content)
                try:
                    parsed_result = json.loads(json_content)
                except json.JSONDecodeError as e:
                    print("解析 JSON 时出错:", e)
                    return None
                return parsed_result

            else:
                print("Error:", result.stderr)
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
