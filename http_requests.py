import requests
import sys

sites = ["http://www.walla.co.il:80", "http://www.ynet.co.il:80", "http://13news.co.il:80",
         "http://www.mako.co.il:80", "http://www.israelhayom.co.il:80", "http://www.n12.co.il:80",
         "http://www.google.com:80", "http://bing.com:80", "http://www.yandex.com:80",
         "http://www.swisscows.com:80", "http://search.creativecommons.org/:80", "http://duckduckgo.com:80"]

user_agent = {'User-agent': 'AI_IDS'}

values = {'name': 'AI_IDS project',
          'location': 'Technion',
          'language': 'Python'}

is_exploit = sys.argv[1]
if is_exploit == "True":
    is_exploit = True
elif is_exploit == "False":
    is_exploit = False
print(is_exploit)

if is_exploit:
    for s in sites:
        try:
            requests.post(s, headers=user_agent, data=values)
        except requests.exceptions.RequestException:
            print("Couldn't connect to: ", s)
else:
    for s in sites:
        try:
            requests.get(s)
        except requests.exceptions.RequestException:
            print("Couldn't connect to: ", s)
