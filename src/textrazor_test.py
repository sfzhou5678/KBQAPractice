import textrazor
textrazor.api_key = "b061d4d2c7fade52e7ae8c3c786eb4c50eb935876aae037159de827e"

client = textrazor.TextRazor(extractors=["entities", "topics",'relations',
                                         # 'words','phrases','meaning',
                                         ])
response = client.analyze("what does jamaican people speak?")

print(response)