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
    TextRazorManager.update_api_key(api_key)
    return TextRazorManager.get_client()

  @staticmethod
  def update_api_key(api_key):
    TextRazorManager.api_key = api_key
    textrazor.api_key = api_key
    TextRazorManager.client = textrazor.TextRazor(extractors=['customAnnotations', 'coarseTopics', 'entailments',
                                                              'properties', 'nounPhrases', 'sentences', 'categories',
                                                              "entities", "topics", 'relations',
                                                              ])
