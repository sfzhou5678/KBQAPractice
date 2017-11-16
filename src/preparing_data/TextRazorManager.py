import textrazor


class TextRazorManager:
  api_key = ''
  client = None

  @staticmethod
  def get_client():
    if TextRazorManager.client is not None:
      return TextRazorManager.client
    else:
      return None

  @staticmethod
  def get_client_with_key(api_key):
    TextRazorManager._update_api_key(api_key)
    return TextRazorManager.get_client()

  @staticmethod
  def _update_api_key(api_key):
    TextRazorManager.api_key = api_key
    textrazor.api_key = api_key
    TextRazorManager.client = textrazor.TextRazor(extractors=['customAnnotations', 'coarseTopics', 'entailments',
                                                              'properties', 'nounPhrases', 'sentences', 'categories',
                                                              "entities", "topics", 'relations',
                                                              ])

  @classmethod
  def get_new_client(cls, api_key_list,old_client):
    if old_client.api_key!=TextRazorManager.api_key:
      return TextRazorManager.client

    if len(api_key_list)>0:
      api_key_list.remove(api_key_list[0])
      old_api_key = TextRazorManager.api_key
      api_key_list.append(old_api_key)  # 为避免网络故障的低效方法：循环利用APIKEY
      api_key = api_key_list[0]
      print('[Change %s -> %s]' % (old_api_key, api_key))
      client = TextRazorManager.get_client_with_key(api_key)

    return client
