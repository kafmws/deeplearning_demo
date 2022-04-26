import urllib
import easyocr
import base64
import cv2
from msedge.selenium_tools import EdgeOptions
from msedge.selenium_tools import Edge

from selenium import webdriver
import os, sys
import time


def initialize_browser():
    global driver
	if driver != None:
        return
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option("excludeSwitches", ['enable-automation', 'ignore-certificate-errors'])
    chrome_options.add_argument('--profile-directory=Default')
    ## for root user in Linux
    #chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument('--user-data-dir=C:/Users/kafm/AppData/Local/Google/Chrome/User Data/Default')
    # for headless mode
    # chrome_options.add_argument('headless')
    driver = webdriver.Chrome(options=chrome_options)

    # edge_options = EdgeOptions()
    # edge_options.add_argument('headless')
    # edge_options.use_chromium = True
    # edge_options.add_experimental_option('useAutomationExtension', False)
    # edge_options.add_experimental_option('excludeSwitches',
    #                                      ['enable-automation', 'enable-logging'])
    # global driver
    # driver = Edge(options=edge_options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                  """
    })
    driver.get('https://ai.baidu.com/tech/imagerecognition/animal')
    time.sleep(5)


def img2base64(img_path: str) -> str:
    with open(img_path, 'rb') as f:
        return urllib.parse.quote(base64.b64encode(f.read()).decode('utf-8'))


def request_script(img: str) -> str:
    return f"""
        fetch("https://ai.baidu.com/aidemo", {{"headers": {{
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "content-type": "application/x-www-form-urlencoded",
            "sec-ch-ua": "\\" Not A;Brand\\";v=\\"99\\", \\"Chromium\\";v=\\"99\\", \\"Microsoft Edge\\";v=\\"99\\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\\"Windows\\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
            }},
            "referrer": "https://ai.baidu.com/tech/imagerecognition/animal",
            "referrerPolicy": "strict-origin-when-cross-origin",
            "body": "image=data%3Aimage%2Fjpg%3Bbase64%2C{img}&image_url&type=animal&show=true",
            "method": "POST",
            "mode": "cors",
            "credentials": "include"
        }})
        .then(function(resp){{return resp.json();}})
        .then(function(data){{
            var p = document.createElement("p");
            p.innerText = JSON.stringify(data);
            p.id = 'hack';
            document.body.appendChild(p);
        }});
        """


def identity(img_path: str) -> str:
    global driver
    if driver is None:
        initialize_browser()
    script = request_script(img2base64(img_path))
    driver.execute_script(script)
    time.sleep(2)
    res = driver.execute_script("""
        node = document.getElementById('hack');
        if (node != null) {
            return node.innerText;
            //document.body.removeChild(node);
        } else { return null; }
    """)
    cnt = 0
    while res is None and cnt < 3:
        time.sleep(1)
        res = driver.execute_script("""
                node = document.getElementById('hack');
                if (node != null) {
                    return node.innerText;
                    //document.body.removeChild(node);
                } else { return null; }
            """)
        cnt += 1
    if only_name:
       return json.loads(res)['data']['result'][0]['name']
    return res


driver = None
if __name__ == '__main__':
    url = 'https://ai.baidu.com/tech/imagerecognition/animal'
    pic_path = '/data/mystack/sites/animalmonitor.cn/www/linye-pc/python/Recognize/test/Antelope0009.jpg'
    parser = argparse.ArgumentParser(description='train CNN')
    parser.add_argument('--pic_path', default='/data/mystack/sites/animalmonitor.cn/www/linye-pc/python/Recognize/test/Antelope0009.jpg', type=str, required=False,
                        help='pic path')
    args = parser.parse_args()
    pic_path = args.pic_path
    result = identity(pic_path, True)
    print(result)
    # result = {'status': 'success', 'data': {'result': [{'score': 0.99, 'class': 'cat', 'class_id': 0, 'class_name': '猫'}]}}
    # import json
    # print(json.dumps(result))